
package com.devanshu.cyberguardai

import android.content.Context
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import java.util.Locale

class UrlDetector(context: Context) {

    private val vocabulary = HashMap<String, Int>()
    private lateinit var modelIdf: DoubleArray
    private lateinit var modelWeights: DoubleArray
    private var intercept = 0.0

    init {
        val json = context.assets.open("url_model.json")
            .bufferedReader()
            .use { it.readText() }

        val obj = JSONObject(json)

        val vocab = obj.getJSONArray("vocabulary")
        val idf = obj.getJSONArray("idf")
        val weights = obj.getJSONArray("weights")

        for (i in 0 until vocab.length()) {
            vocabulary[vocab.getString(i)] = i
        }

        modelIdf = DoubleArray(idf.length()) { idf.getDouble(it) }
        modelWeights = DoubleArray(weights.length()) { weights.getDouble(it) }
        intercept = obj.getDouble("intercept")
    }

    fun analyze(url: String): UrlResult {

        val text = url.lowercase(Locale.ROOT)
        val termFrequency = HashMap<String, Int>()

        // Character n-grams of lengths 3, 4 and 5
        for (n in 3..5) {
            if (text.length >= n) {
                for (i in 0..text.length - n) {
                    val gram = text.substring(i, i + n)
                    termFrequency[gram] =
                        (termFrequency[gram] ?: 0) + 1
                }
            }
        }

        val vector = HashMap<Int, Double>()

        for ((gram, frequency) in termFrequency) {
            val index = vocabulary[gram] ?: continue
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
            probability >= 0.75 -> "Potential Phishing"
            probability <= 0.25 -> "Website Looks Normal"
            else -> "Suspicious Website"
        }

        return UrlResult(classification, probability)
    }
}

data class UrlResult(
    val classification: String,
    val phishingProbability: Double
)
