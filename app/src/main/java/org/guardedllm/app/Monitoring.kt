package org.guardedllm.app

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Text is never persisted. Consent and monitoring state deliberately reset on process death. */
object Monitoring {
    enum class Mode { STOPPED, ACTIVE, PAUSED }
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val classifierLock = Mutex()
    private var classifier: OnnxToxicityClassifier? = null
    private val seen = LinkedHashSet<Int>()
    private val pending = ArrayDeque<String>()
    private var generation = 0L
    private var busy = false
    val mode = MutableStateFlow(Mode.STOPPED)
    val status = MutableStateFlow("Monitoring is off.")
    private val _result = MutableStateFlow<Classification?>(null)
    val result = _result.asStateFlow()

    @Synchronized fun setMode(value: Mode) {
        generation++
        mode.value = value
        seen.clear()
        pending.clear()
        _result.value = null
        status.value = when (value) { Mode.ACTIVE -> "Monitoring requested. Enable the Android accessibility service, then open a supported Instagram chat."; Mode.PAUSED -> "Monitoring paused."; Mode.STOPPED -> "Monitoring stopped." }
    }
    fun analyzeManual(context: Context, text: String) {
        if (text.isBlank()) { status.value = "Enter some text first."; return }
        scope.launch {
            try {
                val output = classify(context, text.take(4096))
                _result.value = output
                status.value = message(output)
            } catch (_: Exception) { status.value = "Model unavailable. Prepare the model assets and rebuild the app." }
        }
    }
    @Synchronized fun submitVisible(context: Context, text: String) {
        if (mode.value != Mode.ACTIVE || text.isBlank()) return
        val bounded = text.take(4096)
        val hash = bounded.hashCode()
        if (hash in seen || pending.size >= 20) return
        seen.add(hash)
        if (seen.size > 128) seen.remove(seen.first())
        pending.addLast(bounded)
        if (!busy) {
            busy = true
            scope.launch { drain(context.applicationContext) }
        }
    }
    private suspend fun classify(context: Context, text: String): Classification = classifierLock.withLock {
        val model = classifier ?: OnnxToxicityClassifier(context.applicationContext).also { classifier = it }
        model.classify(text)
    }
    private suspend fun drain(context: Context) {
        while (true) {
            val next = synchronized(this) {
                if (mode.value != Mode.ACTIVE || pending.isEmpty()) { busy = false; null }
                else Pair(generation, pending.removeFirst())
            } ?: return
            try {
                val output = classify(context, next.second)
                synchronized(this) {
                    if (generation == next.first && mode.value == Mode.ACTIVE) {
                        _result.value = output
                        status.value = message(output)
                        if (output.flagged) Alerts.show(context)
                    }
                }
            } catch (_: Exception) {
                synchronized(this) {
                    if (generation == next.first) {
                        pending.clear()
                        status.value = "Model unavailable. Prepare the model assets and rebuild the app."
                    }
                }
            }
        }
    }
    private fun message(output: Classification) = if (output.flagged) "Potentially harmful language detected. Review the context." else "No alert from the toxicity model. This does not mean the conversation is safe."
}
