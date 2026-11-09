
package com.devanshu.cyberguardai

import android.content.Context
import org.json.JSONObject
import java.util.Locale
import java.util.regex.Pattern
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

class SmsDetector(context: Context) {

    private val vocabulary = HashMap<String, Int>()
    private lateinit var modelIdf: DoubleArray
    private lateinit var modelWeights: DoubleArray
    private var intercept = 0.0

    init {
        val json = context.assets.open("sms_model.json")
            .bufferedReader()
            .use { it.readText() }

        val obj = JSONObject(json)

        val vocabArray = obj.getJSONArray("vocabulary")
        val idfArray = obj.getJSONArray("idf")
        val weightArray = obj.getJSONArray("weights")

        for (i in 0 until vocabArray.length()) {
            vocabulary[vocabArray.getString(i)] = i
        }

        modelIdf = DoubleArray(idfArray.length()) { idfArray.getDouble(it) }
        modelWeights = DoubleArray(weightArray.length()) { weightArray.getDouble(it) }

        intercept = obj.getDouble("intercept")
    }


    fun analyze(message: String): SmsResult {
        val words = Regex("[\\p{L}\\p{N}_]{2,}")
            .findAll(message.lowercase(Locale.ROOT))
            .map { it.value }
            .toList()

        val termFrequency = HashMap<String, Int>()

        for (word in words) {
            termFrequency[word] = (termFrequency[word] ?: 0) + 1
        }

        for (i in 0 until words.size - 1) {
            val bigram = "${words[i]} ${words[i + 1]}"
            termFrequency[bigram] = (termFrequency[bigram] ?: 0) + 1
        }

        val vector = HashMap<Int, Double>()

        for ((term, frequency) in termFrequency) {
            val index = vocabulary[term] ?: continue
            val tf = 1.0 + ln(frequency.toDouble())
            vector[index] = tf * modelIdf[index]
        }

        val norm = sqrt(vector.values.sumOf { it * it })
        var score = intercept

        if (norm > 0.0) {
            for ((index, value) in vector) {
                score += (value / norm) * modelWeights[index]
            }
        }

        val probability = if (score >= 0) {
            1.0 / (1.0 + exp(-score))
        } else {
            val e = exp(score)
            e / (1.0 + e)
        }

        val classification = when {
            probability >= 0.75 -> "Potential Scam"
            probability <= 0.25 -> "Likely Safe"
            else -> "Suspicious"
        }

        return SmsResult(classification, probability)
    }

}

data class SmsResult(
    val classification: String,
    val spamProbability: Double
)
