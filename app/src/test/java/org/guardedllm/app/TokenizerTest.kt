package org.guardedllm.app

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.util.Base64
import org.junit.Assume.assumeNotNull

class TokenizerTest {
    private val tokenizer = BertTokenizer(listOf("[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", ",", "world", "un", "##kind", "!"))
    @Test fun tokenizesUncasedPunctuationAndWordpieces() {
        assertArrayEquals(longArrayOf(2, 4, 5, 6, 7, 8, 9, 3), tokenizer.encode("Héllo, world unkind!"))
    }
    @Test fun matchesHuggingFaceReferenceVectors() {
        val vocabStream = java.io.File("src/main/assets/model/vocab.txt")
        val vectors = java.io.File("src/test/resources/tokenizer-vectors.tsv")
        assumeNotNull(vocabStream.takeIf { it.exists() }, vectors.takeIf { it.exists() })
        val actualTokenizer = BertTokenizer(vocabStream.readLines())
        vectors.readLines().forEach { line ->
            val (encoded, ids) = line.split('\t')
            val text = String(Base64.getDecoder().decode(encoded), Charsets.UTF_8)
            assertArrayEquals(ids.split(',').map { it.toLong() }.toLongArray(), actualTokenizer.encode(text))
        }
    }
    @Test fun unknownWordUsesUnknownToken() {
        assertArrayEquals(longArrayOf(2, 1, 3), tokenizer.encode("something"))
    }
}
