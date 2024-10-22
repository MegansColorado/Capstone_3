�	�\5϶W@�\5϶W@!�\5϶W@	�Q���?�Q���?!�Q���?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�\5϶W@k}�Жs�?18,��)W@I,���o�?Y�ɋL���?*	�n��b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��
���?!���#��B@)w����?1�����9@:Preprocessing2U
Iterator::Model::ParallelMapV2���E��?!k�4���0@)���E��?1k�4���0@:Preprocessing2F
Iterator::Model�V�I��?!.�c�)?@)�����?1�'ʿ&-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�����?!��Ʃ2@)Tq��s�?1�L�m�;*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�p̲'�?!���="'@){�p̲'�?1���="'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�`6��?!u�9'�5Q@)KZ��φ?1�����@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceHk:!t�?!��00@)Hk:!t�?1��00@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�)���?!n0r��7@)��)�~?1�u�u�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�Q���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	k}�Жs�?k}�Жs�?!k}�Жs�?      ��!       "	8,��)W@8,��)W@!8,��)W@*      ��!       2      ��!       :	,���o�?,���o�?!,���o�?B      ��!       J	�ɋL���?�ɋL���?!�ɋL���?R      ��!       Z	�ɋL���?�ɋL���?!�ɋL���?JGPUY�Q���?b �"{
Qgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter[�G��?![�G��?"_
Cgradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2DConv2D���Mh�?!Csd��`�?"W
.functional_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput�|���@�?!xR -1�?"i
?gradient_tape/functional_1/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�9�qe]�?!l�D6$�?"7
functional_1/dense/BiasAddBiasAdd��if(�?!��\����?"g
Jfunctional_1/dense/BiasAdd-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown�'�Q�?!y[��{�?"�
ugradient_tape/functional_1/conv2d_transpose/conv2d_transpose/Conv2D-0-1-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknownDG=��*�?!�/C����?"T
8gradient_tape/functional_1/dense/Tensordot/MatMul/MatMulMatMul�(�[��?!1!����?"<
functional_1/conv2d_1/Relu_FusedConv2D ߳��o�?!)�����?"?
#functional_1/dense/Tensordot/MatMulMatMul8
xX+Ɨ?!{p�UFh�?Q      Y@Y����=1@a��=��T@q�fv6p@yñk��#m?"�
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: B 