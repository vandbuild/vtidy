/**
 * Copyright 2023 Viacheslav Barkov
 * Optimized for Large Image Datasets in 2026
 */

package com.slavabarkov.tidy.viewmodels

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.app.Application
import android.content.ContentResolver
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import androidx.lifecycle.*
import com.slavabarkov.tidy.R
import com.slavabarkov.tidy.centerCrop
import com.slavabarkov.tidy.data.ImageEmbedding
import com.slavabarkov.tidy.data.ImageEmbeddingDatabase
import com.slavabarkov.tidy.data.ImageEmbeddingRepository
import com.slavabarkov.tidy.normalizeL2
import com.slavabarkov.tidy.preProcess
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.*

class ORTImageViewModel(application: Application) : AndroidViewModel(application) {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var repository: ImageEmbeddingRepository
    var idxList: ArrayList<Long> = arrayListOf()
    var embeddingsList: ArrayList<FloatArray> = arrayListOf()
    var progress: MutableLiveData<Double> = MutableLiveData(0.0)

    init {
        val imageEmbeddingDao = ImageEmbeddingDatabase.getDatabase(application).imageEmbeddingDao()
        repository = ImageEmbeddingRepository(imageEmbeddingDao)
    }

    fun generateIndex() {
        val modelID = R.raw.visual_quant
        val resources = getApplication<Application>().resources
        val model = resources.openRawResource(modelID).readBytes()
        val session = ortEnv.createSession(model)

        // फिक्स: ऐप क्रैश से बचाने के लिए भारी काम को बैकग्राउंड थ्रेड (Dispatchers.IO) पर डाला
        viewModelScope.launch(Dispatchers.IO) {
            progress.postValue(0.0)
            val uri: Uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            val projection = arrayOf(
                MediaStore.Images.Media._ID,
                MediaStore.Images.Media.DATE_MODIFIED,
                MediaStore.Images.Media.BUCKET_DISPLAY_NAME
            )
            val sortOrder = "${MediaStore.Images.Media._ID} ASC"
            val contentResolver: ContentResolver = getApplication<Application>().contentResolver
            val cursor: Cursor? = contentResolver.query(uri, projection, null, null, sortOrder)
            val totalImages = cursor?.count ?: 0
            
            cursor?.use {
                val idColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val dateColumn: Int =
                    it.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
                val bucketColumn: Int =
                    it.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME)
                
                while (it.moveToNext()) {
                    val id: Long = it.getLong(idColumn)
                    val date: Long = it.getLong(dateColumn)
                    val bucket: String = it.getString(bucketColumn)
                    
                    // स्क्रीनशॉट्स को इंडेक्स न करें
                    if (bucket == "Screenshots") continue
                    
                    val record = repository.getRecord(id) as ImageEmbedding?
                    if (record != null) {
                        idxList.add(record.id)
                        embeddingsList.add(record.embedding)
                    } else {
                        val imageUri: Uri = Uri.withAppendedPath(uri, id.toString())
                        
                        try {
                            // use block का उपयोग किया ताकि InputStream काम के बाद अपने आप बंद हो जाए
                            contentResolver.openInputStream(imageUri)?.use { inputStream ->
                                val bytes = inputStream.readBytes()
                                
                                // बिटमैप डिकोड करें
                                val bitmap: Bitmap? =
                                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                                    
                                bitmap?.let { sourceBitmap ->
                                    val rawBitmap = centerCrop(sourceBitmap, 224)
                                    val inputShape = longArrayOf(1, 3, 224, 224)
                                    val inputName = "pixel_values"
                                    val imgData = preProcess(rawBitmap)
                                    val inputTensor = OnnxTensor.createTensor(ortEnv, imgData, inputShape)

                                    inputTensor.use {
                                        val output =
                                            session?.run(Collections.singletonMap(inputName, inputTensor))
                                        output.use {
                                            @Suppress("UNCHECKED_CAST") 
                                            var rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                                            rawOutput = normalizeL2(rawOutput)
                                            repository.addImageEmbedding(
                                                ImageEmbedding(
                                                    id, date, rawOutput
                                                )
                                            )
                                            idxList.add(id)
                                            embeddingsList.add(rawOutput)
                                        }
                                    }
                                    
                                    // फिक्स: रैम को खाली करने के लिए बिटमैप्स को तुरंत रीसायकल (Recycle) किया
                                    rawBitmap.recycle()
                                    if (sourceBitmap != rawBitmap) {
                                        sourceBitmap.recycle()
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            // अगर कोई फोटो करप्ट है, तो ऐप क्रैश होने के बजाय उसे स्किप कर देगा
                            e.printStackTrace() 
                        }
                    }
                    // प्रोग्रेस को सुरक्षित रूप से बैकग्राउंड थ्रेड से अपडेट करने के लिए postValue लगाया
                    progress.postValue(it.position.toDouble() / totalImages.toDouble())
                }
            }
            cursor?.close()
            session.close()
            progress.postValue(1.0)
        }
    }
}
