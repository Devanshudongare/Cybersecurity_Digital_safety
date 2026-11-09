package com.devanshu.cyberguardai

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "scan_history")
data class ScanRecord(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val analyzer: String,
    val inputData: String,
    val resultLabel: String,
    val riskScore: Double,
    val timestamp: Long = System.currentTimeMillis()
)
