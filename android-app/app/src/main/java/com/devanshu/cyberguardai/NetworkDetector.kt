
package com.devanshu.cyberguardai

import android.content.Context
import org.json.JSONObject
import org.json.JSONArray
import kotlin.math.roundToInt

data class NetworkResult(
    val classification: String,
    val suspiciousProbability: Double
)

class NetworkDetector(context: Context) {

    private val model: JSONObject =
        JSONObject(
            context.assets.open("network_model.json")
                .bufferedReader()
                .use { it.readText() }
        )

    private val features: JSONArray = model.getJSONArray("features")
    private val medians: JSONArray = model.getJSONArray("medians")
    private val trees: JSONArray = model.getJSONArray("trees")

    fun analyze(input: Map<String, Double?>): NetworkResult {

        val featureValues = DoubleArray(features.length())

        // Prepare all 68 features and replace missing values with medians
        for (i in 0 until features.length()) {
            val name = features.getString(i)
            val value = input[name]

            featureValues[i] =
                if (value == null || value.isNaN() || value.isInfinite()) {
                    medians.getDouble(i)
                } else {
                    value
                }
        }

        var totalSuspiciousProbability = 0.0

        // Run all 100 Random Forest trees
        for (treeIndex in 0 until trees.length()) {
            val tree = trees.getJSONObject(treeIndex)

            val left = tree.getJSONArray("left")
            val right = tree.getJSONArray("right")
            val feature = tree.getJSONArray("feature")
            val threshold = tree.getJSONArray("threshold")
            val probabilities = tree.getJSONArray("probabilities")

            var node = 0

            // Traverse until a leaf is reached
            while (left.getInt(node) != -1 ){
                    val featureIndex = feature.getInt(node)
                    val splitValue = threshold.getDouble(node)

                    node = if (featureValues[featureIndex] <= splitValue) {
                        left.getInt(node)
                    } else {
                        right.getInt(node)
                    }
                }

                // Class index 1 = suspicious
                totalSuspiciousProbability +=
                probabilities.getJSONArray(node).getDouble(1)
        }

        val probability =
            totalSuspiciousProbability / trees.length()

        val result = when {
            probability >= 0.75 -> "Potential Threat"
            probability <= 0.25 -> "Likely Normal Traffic"
            else -> "Suspicious Traffic"
        }

        return NetworkResult(
            classification = result,
            suspiciousProbability = probability
        )
    }

    fun getFeatureNames(): List<String> {
        return (0 until features.length()).map {
            features.getString(it)
        }
    }
}
