�	�g$B#Wk@�g$B#Wk@!�g$B#Wk@	������?������?!������?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�g$B#Wk@�C�3��?1ӿ$�)�j@I}��b9 @YL�
F%u�?*	fffffa@2F
Iterator::ModelY32�]��?!f���:B@)<FzQ��?1�&��*�6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�b����?!���f=@)y;�i���?1���GN�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�9��?!1�R��6@)ߣ�z��?1��'M<r0@:Preprocessing2U
Iterator::Model::ParallelMapV2vOjM�?!�Jw�!�+@)vOjM�?1�Jw�!�+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip �C��<�?!�FQb�O@)ȳ˷>��?1��W� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�:�6�?!���MO@)�:�6�?1���MO@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���H��?!�ʪ1�@)���H��?1�ʪ1�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_�2���?!��0��9@)�B�Գ t?1ET�^�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9������?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�C�3��?�C�3��?!�C�3��?      ��!       "	ӿ$�)�j@ӿ$�)�j@!ӿ$�)�j@*      ��!       2      ��!       :	}��b9 @}��b9 @!}��b9 @B      ��!       J	L�
F%u�?L�
F%u�?!L�
F%u�?R      ��!       Z	L�
F%u�?L�
F%u�?!L�
F%u�?JGPUY������?b �"{
Qgradient_tape/sequential/conv2d_transpose_2/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter������?!������?"_
Cgradient_tape/sequential/conv2d_transpose_2/conv2d_transpose/Conv2DConv2D�d�Q�?!G}�Z��?"W
.sequential/conv2d_transpose_2/conv2d_transposeConv2DBackpropInputKS���¥?!��&�?"W
.sequential/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput� ��ꖞ?!,r	=���?"g
=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter��A>]�?!ג��Db�?"e
<gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInputc��s�:�?!��ՙ�E�?"g
=gradient_tape/sequential/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�d��0�?!'hp��'�?"G
)gradient_tape/sequential/dense_1/ReluGradReluGrad���C��?!�����?"{
Qgradient_tape/sequential/conv2d_transpose_3/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�*�r�?!.�K���?"G
)gradient_tape/sequential/dense_2/ReluGradReluGrad�X�dr�?!�����?Q      Y@Y��8��8&@a��8��8V@qH��G�B@y((o0"�Y?"�	
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�37.1116% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 