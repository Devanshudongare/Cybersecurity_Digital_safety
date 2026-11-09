package com.devanshu.cyberguardai

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "learning_records")
data class LearningRecord(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val analyzer: String,
    val inputData: String,
    val predictedLabel: String,
    val verifiedLabel: String,
    val confidence: Double,
    val timestamp: Long = System.currentTimeMillis()
)
