�	 8���_@ 8���_@! 8���_@	�pi��v�?�pi��v�?!�pi��v�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6 8���_@3����=@1[��	X@A�3w�ɧ?I�!p$P�?Y�[���?*	k�t�a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$�@�?!�	,�N;@)��Co��?1����'6@:Preprocessing2F
Iterator::Model��t�i��?!�����A@)����ם?1s�/K�^5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeati5$���?!	�U�9@)K�.��"�?1o��a�2@:Preprocessing2U
Iterator::Model::ParallelMapV2��G�)s�?!2g ���+@)��G�)s�?12g ���+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��*Q���?!���-P@)`=�[��?1�uP� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3�`��?!h�O�g @)3�`��?1h�O�g @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�?�|?!UbpE��@)�?�|?1UbpE��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap=e5]Ot�?!�]*�¹>@)A�vs?1����X@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 23.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�pi��v�?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	3����=@3����=@!3����=@      ��!       "	[��	X@[��	X@![��	X@*      ��!       2	�3w�ɧ?�3w�ɧ?!�3w�ɧ?:	�!p$P�?�!p$P�?!�!p$P�?B      ��!       J	�[���?�[���?!�[���?R      ��!       Z	�[���?�[���?!�[���?JGPUY�pi��v�?b �"~
Tgradient_tape/functional_5/conv2d_transpose_16/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilteribد���?!ibد���?"b
Fgradient_tape/functional_5/conv2d_transpose_16/conv2d_transpose/Conv2DConv2D�)BvW�?!(F�����?"Z
1functional_5/conv2d_transpose_16/conv2d_transposeConv2DBackpropInput�&a���?!#�?���?"j
@gradient_tape/functional_5/conv2d_14/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�a(�Kk�?!��$a)o�?"9
functional_5/dense_8/BiasAddBiasAdd����ZƬ?!���;�?"i
Lfunctional_5/dense_8/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown �F\��?!������?"�
xgradient_tape/functional_5/conv2d_transpose_16/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown|��p�V�?!���l"�?"=
functional_5/conv2d_14/Relu_FusedConv2Di�&iɝ?!8�յm��?"V
:gradient_tape/functional_5/dense_8/Tensordot/MatMul/MatMulMatMulb����?!H���9��?"h
?gradient_tape/functional_5/conv2d_14/Conv2D/Conv2DBackpropInputConv2DBackpropInput���.�h�?!-Dqs��?Q      Y@Y����=1@a��=��T@q�[0<L@y�m
4�l?"�	
both�Your program is POTENTIALLY input-bound because 23.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�56.4695% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 