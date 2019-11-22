import tensorflow_hub as hub
import keras.backend as K
import keras
from bert import run_classifier, tokenization
import tensorflow as tf

def create_tokenizer_from_hub_module(bert_hub_module_handle):
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(bert_hub_module_handle)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)



def get_bert_inputs(input, bert_hub_module_handle, bert_max_seqlen):
    tokenizer = create_tokenizer_from_hub_module(bert_hub_module_handle)
    features = run_classifier.convert_examples_to_features(input, [0, 1], bert_max_seqlen, tokenizer)
    input_ids = []
    input_mask = []
    segment_ids = []
    for feature in features:
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)

    return input_ids, input_mask, segment_ids

class BertLayer(keras.layers.Layer):
    def __init__(self, bert_hub_module_handle, n_fine_tune_layers=10, **kwargs):
        self.bert_hub_module_handle = bert_hub_module_handle
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_hub_module_handle,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)["pooled_output"]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)
