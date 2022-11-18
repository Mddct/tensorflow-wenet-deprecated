from typing import List, Tuple
import tensorflow as tf


class CTCSearch(tf.Module):

    def __init__(self, beam_size, blank_index=0, dtype=tf.float32):
        super().__init__(name="CTCSearch")
        self.blank_index = blank_index
        self.beam_size = beam_size * 2
        self.top_size = beam_size
        self.dtype = tf.as_dtype(dtype)

    def greedy_search(self, logits,
                      logits_lens) -> Tuple[tf.RaggedTensor, tf.Tensor]:
        """CTC greedy serch for sequences with higest scores

        Args:
           logits: [batch_size, time, vocab_size]
           logits_lens: [B], int32 vector containing logits lengths

        Returns:

        """
        logits = tf.transpose(logits,
                              [1, 0, 2])  #[time, batch_size, vocab_size]
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            logits,
            logits_lens,
            blank_index=self.blank_index,
            merge_repeated=True,
        )
        return tf.RaggedTensor.from_sparse(decoded), neg_sum_logits

    def beam_search(self, logits,
                    logits_lens) -> Tuple[tf.RaggedTensor, tf.Tensor]:
        """CTC beam serch for sequences with higest scores
        Args:
           logits: [batch_size, time, vocab_size]
           logits_lens: [B], int32 vector containing logits lengths

        Returns:
        """

        logits = tf.transpose(logits,
                              [1, 0, 2])  #[time, batch_size, vocab_size]

        decoded, neg_sum_logits = tf.nn.ctc_beam_search_decoder(
            logits,
            logits_lens,
            beam_width=self.beam_size,
            top_paths=self.top_size)
        return tf.RaggedTensor.from_sparse(decoded), neg_sum_logits
