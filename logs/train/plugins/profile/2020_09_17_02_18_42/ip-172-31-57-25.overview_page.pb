�	ߊ�ӂ@ߊ�ӂ@!ߊ�ӂ@	�̗�U�?�̗�U�?!�̗�U�?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ߊ�ӂ@JV�˯@1V�Zޣ�@IIg`�e�@Y)�1k��?*	(1�2b@2F
Iterator::Model�
�.Ȯ?![�}ޘ�D@)t34��?1�&dZS28@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�|^�ԣ?!V�{���:@)� Pō[�?1�ydv<3@:Preprocessing2U
Iterator::Model::ParallelMapV2�Ϛ�?!��b�1@)�Ϛ�?1��b�1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-#���i�?!@��5@)�y��C5�?1�̪�t,@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�2R臭�?!�N]��V@)�2R臭�?1�N]��V@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip R�8�ߵ?!�e�!gYM@)������?1�A���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice.S��i�?!�P�*�b@).S��i�?1�P�*�b@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapmY�.��?!�m�/�8@)�>��Vv?1�B�/Z�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�̗�U�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	JV�˯@JV�˯@!JV�˯@      ��!       "	V�Zޣ�@V�Zޣ�@!V�Zޣ�@*      ��!       2      ��!       :	Ig`�e�@Ig`�e�@!Ig`�e�@B      ��!       J	)�1k��?)�1k��?!)�1k��?R      ��!       Z	)�1k��?)�1k��?!)�1k��?JGPUY�̗�U�?b �"}
Sgradient_tape/functional_5/conv2d_transpose_6/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterR;֪p�?!R;֪p�?"a
Egradient_tape/functional_5/conv2d_transpose_6/conv2d_transpose/Conv2DConv2D6�z׾��?!���A���?"Y
0functional_5/conv2d_transpose_6/conv2d_transposeConv2DBackpropInputr�bc���?!{�h�Y�?"i
?gradient_tape/functional_5/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��
���?!mJ	W��?"]
<gradient_tape/functional_5/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad�u�S"�?!&p�i��?"T
6gradient_tape/functional_5/conv2d_transpose_6/ReluGradReluGradqu�ޔ�?!Ԟ
I=4�?"Y
0functional_5/conv2d_transpose_7/conv2d_transposeConv2DBackpropInpute��E�h�?!��qUa�?"}
Sgradient_tape/functional_5/conv2d_transpose_7/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter$�x0��?!f2�p��?"i
?gradient_tape/functional_5/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��O�cR�?!.@����?"J
,gradient_tape/functional_5/conv2d_4/ReluGradReluGradB����?!������?Q      Y@Yu8�~k@ay\�NW@q�j�!�t>@y�y\[ML?"�	
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�30.4552% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 