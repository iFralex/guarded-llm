package org.guardedllm.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.text.Normalizer
import java.util.Locale
import kotlin.math.exp

interface TextClassifier {
    fun classify(text: String): Classification
}
data class Classification(val scores: Map<String, Float>, val elapsedMs: Long) {
    val flagged: Boolean get() = (scores["toxic"] ?: 0f) >= 0.5f
}

/** BERT uncased basic tokenization and greedy WordPiece, using the model's exact vocabulary. */
class BertTokenizer(vocabulary: List<String>, private val maxLength: Int = 128) {
    private val vocab = vocabulary.withIndex().associate { it.value to it.index.toLong() }
    private fun isPunctuation(c: Char): Boolean {
        val type = Character.getType(c)
        return (c.code in 33..47 || c.code in 58..64 || c.code in 91..96 || c.code in 123..126 ||
            type == Character.CONNECTOR_PUNCTUATION.toInt() || type == Character.DASH_PUNCTUATION.toInt() ||
            type == Character.START_PUNCTUATION.toInt() || type == Character.END_PUNCTUATION.toInt() ||
            type == Character.OTHER_PUNCTUATION.toInt() || type == Character.INITIAL_QUOTE_PUNCTUATION.toInt() || type == Character.FINAL_QUOTE_PUNCTUATION.toInt())
    }
    fun encode(input: String): LongArray {
        val normalized = Normalizer.normalize(input.lowercase(Locale.ROOT), Normalizer.Form.NFD)
            .filter { Character.getType(it) != Character.NON_SPACING_MARK.toInt() && (it == '\t' || it == '\n' || it == '\r' || !Character.isISOControl(it)) }
        val basic = mutableListOf<String>()
        val current = StringBuilder()
        fun flush() { if (current.isNotEmpty()) { basic.add(current.toString()); current.clear() } }
        normalized.forEach { c -> when { c.isWhitespace() -> flush(); isPunctuation(c) -> { flush(); basic.add(c.toString()) }; else -> current.append(c) } }
        flush()
        val ids = mutableListOf(vocab.getValue("[CLS]"))
        for (word in basic) {
            val pieces = mutableListOf<Long>()
            var start = 0
            if (word.length > 100) pieces.add(vocab.getValue("[UNK]")) else {
                while (start < word.length) {
                    var end = word.length
                    var found: Long? = null
                    while (start < end) {
                        val part = (if (start > 0) "##" else "") + word.substring(start, end)
                        found = vocab[part]
                        if (found != null) break
                        end--
                    }
                    if (found == null) { pieces.clear(); pieces.add(vocab.getValue("[UNK]")); break }
                    pieces.add(found); start = end
                }
            }
            for (piece in pieces) { if (ids.size >= maxLength - 1) break; ids.add(piece) }
            if (ids.size >= maxLength - 1) break
        }
        ids.add(vocab.getValue("[SEP]"))
        return ids.toLongArray()
    }
}

class OnnxToxicityClassifier(context: Context) : TextClassifier, AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val tokenizer = BertTokenizer(context.assets.open("model/vocab.txt").bufferedReader().use { it.readLines() })
    private val session: OrtSession = run {
        // Stage model bytes, never conversation data, to avoid loading the entire model into a Java byte array.
        val modelFile = File(context.cacheDir, "toxic-bert.onnx")
        val temporary = File(context.cacheDir, "toxic-bert.onnx.tmp")
        context.assets.open("model/toxic-bert.onnx").use { input ->
            temporary.outputStream().use { output -> input.copyTo(output) }
        }
        check(temporary.renameTo(modelFile)) { "Cannot prepare model file" }
        OrtSession.SessionOptions().use { options -> env.createSession(modelFile.absolutePath, options) }
    }
    private val labels = listOf("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    @Synchronized override fun classify(text: String): Classification {
        val started = System.nanoTime()
        val ids = tokenizer.encode(text)
        val mask = LongArray(ids.size) { 1L }
        val types = LongArray(ids.size)
        val tensors = mapOf(
            "input_ids" to OnnxTensor.createTensor(env, arrayOf(ids)),
            "attention_mask" to OnnxTensor.createTensor(env, arrayOf(mask)),
            "token_type_ids" to OnnxTensor.createTensor(env, arrayOf(types))
        )
        try {
            session.run(tensors.filterKeys { it in session.inputNames }).use { result ->
                @Suppress("UNCHECKED_CAST") val logits = (result[0].value as Array<FloatArray>)[0]
                require(logits.size == labels.size) { "Unexpected model output" }
                return Classification(labels.zip(logits.map { (1.0 / (1.0 + exp(-it.toDouble()))).toFloat() }).toMap(), (System.nanoTime() - started) / 1_000_000)
            }
        } finally { tensors.values.forEach { it.close() } }
    }
    override fun close() = session.close()
}
