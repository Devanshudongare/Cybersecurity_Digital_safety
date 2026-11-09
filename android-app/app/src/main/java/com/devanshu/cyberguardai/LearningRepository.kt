package com.devanshu.cyberguardai

import android.content.Context

class LearningRepository(context: Context) {

    private val dao =
        LearningDatabase.getDatabase(context).learningDao()

    // Feedback
    suspend fun saveVerifiedExample(
        analyzer: String,
        input: String,
        predicted: String,
        verified: String,
        confidence: Double
    ) {
        dao.insert(
            LearningRecord(
                analyzer = analyzer,
                inputData = input,
                predictedLabel = predicted,
                verifiedLabel = verified,
                confidence = confidence
            )
        )
    }

    suspend fun getAllRecords(): List<LearningRecord> = dao.getAll()
    suspend fun getCount(): Int = dao.count()
    suspend fun deleteRecord(record: LearningRecord) = dao.delete(record)
    suspend fun deleteAllRecords() = dao.deleteAll()

    // Scan history
    suspend fun saveScan(
        analyzer: String,
        input: String,
        result: String,
        riskScore: Double
    ) {
        dao.insertScan(
            ScanRecord(
                analyzer = analyzer,
                inputData = input,
                resultLabel = result,
                riskScore = riskScore
            )
        )
    }

    suspend fun getAllScans(): List<ScanRecord> = dao.getAllScans()
    suspend fun getScanCount(): Int = dao.scanCount()

    suspend fun getCountByLabel(label: String): Int =
        dao.countScansByLabel(label)

    suspend fun deleteScan(record: ScanRecord) = dao.deleteScan(record)
    suspend fun deleteAllScans() = dao.deleteAllScans()
}
