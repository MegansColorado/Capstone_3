�	�JY����@�JY����@!�JY����@	u�

���?u�

���?!u�

���?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�JY����@~��7@1C��A|��@I?�G�3F�?Y�)��s�?*	\���($d@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�,D���?!N�=7?@)}ZEh�?1^����5@:Preprocessing2U
Iterator::Model::ParallelMapV2K�����?!R�r�4@)K�����?1R�r�4@:Preprocessing2F
Iterator::Model�H/j���?!:'b5D@)tF��_�?1���#�3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�@+0du�?!RH��O�0@)�uŌ���?1q��B�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���{h�?!H�j�	#@)���{h�?1H�j�	#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=C8fٓ�?!������M@)���{�?1��O�H� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicel��F���?!g��P�p@)l��F���?1g��P�p@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��
/��?!�Zb
4@)"���kv?1����-@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9u�

���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	~��7@~��7@!~��7@      ��!       "	C��A|��@C��A|��@!C��A|��@*      ��!       2      ��!       :	?�G�3F�??�G�3F�?!?�G�3F�?B      ��!       J	�)��s�?�)��s�?!�)��s�?R      ��!       Z	�)��s�?�)��s�?!�)��s�?JGPUYu�

���?b �"{
Qgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter���W�O�?!���W�O�?"_
Cgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DConv2D�C|3(�?!��'.<�?"W
.functional_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput\~'�_��?!n�����?"T
8gradient_tape/functional_1/dense/Tensordot/MatMul/MatMulMatMul���t��?!�� ���?"i
?gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterK�%x�Ĭ?!�C�c6��?"?
#functional_1/dense/Tensordot/MatMulMatMul]^�hf�?!����Z�?"V
:gradient_tape/functional_1/dense/Tensordot/MatMul/MatMul_1MatMul���X��?!��#}��?"7
functional_1/dense/BiasAddBiasAdd��f��?!���f��?"<
functional_1/conv2d_1/Relu_FusedConv2D�֥��?!������?"�
ugradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown�q`�4I�?!S�����?Q      Y@Y�_��_�@aZ�Z�W@q�UF�9�?y�Pʲ�<?"�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 