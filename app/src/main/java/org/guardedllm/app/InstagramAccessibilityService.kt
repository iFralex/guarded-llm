package org.guardedllm.app

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.os.PowerManager
import android.text.InputType

/** Conservative opt-in extraction. Instagram may change or hide its accessibility tree. */
class InstagramAccessibilityService : AccessibilityService() {
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (Monitoring.mode.value != Monitoring.Mode.ACTIVE || event?.packageName?.toString() != "com.instagram.android" || event.isPassword || !getSystemService(PowerManager::class.java).isInteractive) return
        val root = rootInActiveWindow ?: return
        if (root.packageName?.toString() != "com.instagram.android") { root.recycle(); return }
        val candidates = mutableListOf<String>()
        var chatMarker = false
        var visited = 0
        fun walk(node: AccessibilityNodeInfo) {
            if (++visited > 1500 || !node.isVisibleToUser || node.isPassword) return
            val id = node.viewIdResourceName?.substringAfterLast('/')?.lowercase().orEmpty()
            val hint = node.hintText?.toString()?.lowercase().orEmpty()
            if (node.isEditable && (id.contains("composer") || id.contains("message") || hint.contains("message"))) chatMarker = true
            // Only explicitly identified message text nodes; never read editable fields, passwords or generic labels.
            val variation = node.inputType and (InputType.TYPE_MASK_CLASS or InputType.TYPE_MASK_VARIATION)
            val sensitive = variation == (InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD) ||
                variation == (InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD) ||
                variation == (InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD) ||
                variation == (InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_VARIATION_PASSWORD)
            if (!node.isEditable && !sensitive && (id == "message_text" || id == "direct_message_text" || id == "message_content")) {
                node.text?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let(candidates::add)
            }
            for (i in 0 until node.childCount) node.getChild(i)?.let { child -> try { walk(child) } finally { child.recycle() } }
        }
        try { walk(root) } finally { root.recycle() }
        if (chatMarker) candidates.takeLast(20).forEach { Monitoring.submitVisible(applicationContext, it) }
    }
    override fun onInterrupt() { Monitoring.setMode(Monitoring.Mode.PAUSED) }
    override fun onDestroy() { Monitoring.setMode(Monitoring.Mode.STOPPED); super.onDestroy() }
}
