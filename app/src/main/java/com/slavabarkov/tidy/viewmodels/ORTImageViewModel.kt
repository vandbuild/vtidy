/**
 * Copyright 2023 Viacheslav Barkov
 * Optimized for Large Image Datasets in 2026 - Crash & Memory Fix Added
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
import android.util.Log
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
import java.io.InputStream
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

    // भारी इमेज को बिना रैम फुल किए छोटा करके लोड करने का सुरक्षित फंक्शन
    private fun decodeSampledBitmapFromUri(contentResolver: ContentResolver, imageUri: Uri, reqWidth: Int = 224, reqHeight: Int = 224): Bitmap? {
        var inputStream: InputStream? = null
        return try {
            inputStream = contentResolver.openInputStream(imageUri)
            val options = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeStream(inputStream, null, options)
            inputStream?.close()

            var inSampleSize = 1
            if (options.outHeight > reqHeight || options.outWidth > reqWidth) {
                val halfHeight = options.outHeight / 2
                val halfWidth = options.outWidth / 2
                while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                    inSampleSize *= 2
                }
            }

            options.inSampleSize = inSampleSize
            options.inJustDecodeBounds = false

            inputStream = contentResolver.openInputStream(imageUri)
            BitmapFactory.decodeStream(inputStream, null, options)
        } catch (e: Exception) {
            null
        } finally {
            inputStream?.close()
        }
    }

    fun generateIndex() {
        val modelID = R.raw.visual_quant
        val resources = getApplication<Application>().resources
        val model = resources.openRawResource(modelID).readBytes()
        val session = ortEnv.createSession(model)

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
            
            var processedLoopCount = 0

            cursor?.use {
                val idColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
                val dateColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
                val bucketColumn: Int = it.getColumnIndexOrThrow(MediaStore.Images.Media.BUCKET_DISPLAY_NAME)
                
                while (it.moveToNext()) {
                    processedLoopCount++
                    val id: Long = it.getLong(idColumn)
                    val date: Long = it.getLong(dateColumn)
                    val bucket: String = it.getString(bucketColumn)
                    
                    if (bucket == "Screenshots") continue
                    
                    val record = repository.getRecord(id) as ImageEmbedding?
                    if (record != null) {
                        idxList.add(record.id)
                        embeddingsList.add(record.embedding)
                    } else {
                        val imageUri: Uri = Uri.withAppendedPath(uri, id.toString())
                        
                        // ट्राई-कैच का सुरक्षा कवच जो 76% पर क्रैश होने से बचाएगा
                        try {
                            // रैम बचाने के लिए इमेज को रीसाइज़ करके लोड किया
                            val bitmap: Bitmap? = decodeSampledBitmapFromUri(contentResolver, imageUri, 224, 224)
                                    
                            bitmap?.let { sourceBitmap ->
                                val rawBitmap = centerCrop(sourceBitmap, 224)
                                val inputShape = longArrayOf(1, 3, 224, 224)
                                val inputName = "pixel_values"
                                val imgData = preProcess(rawBitmap)
                                val inputTensor = OnnxTensor.createTensor(ortEnv, imgData, inputShape)

                                inputTensor.use {
                                    val output = session?.run(Collections.singletonMap(inputName, inputTensor))
                                    output.use {
                                        @Suppress("UNCHECKED_CAST") 
                                        var rawOutput = ((output?.get(0)?.value) as Array<FloatArray>)[0]
                                        rawOutput = normalizeL2(rawOutput)
                                        repository.addImageEmbedding(
                                            ImageEmbedding(id, date, rawOutput)
                                        )
                                        idxList.add(id)
                                        embeddingsList.add(rawOutput)
                                    }
                                }
                                
                                // बिटमैप मेमोरी को तुरंत रीसायकल करें
                                rawBitmap.recycle()
                                if (sourceBitmap != rawBitmap) {
                                    sourceBitmap.recycle()
                                }
                            }
                        } catch (oom: OutOfMemoryError) {
                            // रैम फुल होने पर तुरंत सफाई
                            System.gc()
                            Runtime.getRuntime().gc()
                            Log.e("TIDY_FIX", "Out of memory handled at photo #$processedLoopCount")
                        } catch (e: Exception) {
                            // 76% पर जो भी करप्ट फोटो आ रही थी, उसे बिना ऐप बंद किए स्किप कर दिया जाएगा
                            Log.e("TIDY_FIX", "Corrupted photo skipped at photo #$processedLoopCount")
                            e.printStackTrace() 
                        }
                    }
                    
                    progress.postValue(it.position.toDouble() / totalImages.toDouble())

                    // हर 20 फोटोज़ प्रोसेस होने के बाद रैम को साफ़ करें ताकि 8000 फोटोज़ आराम से चलें
                    if (processedLoopCount % 20 == 0) {
                        System.gc()
                    }
                }
            }
            cursor?.close()
            session.close()
            progress.postValue(1.0)
        }
    }
}
