�	8�Gn��u@8�Gn��u@!8�Gn��u@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-8�Gn��u@����@1�~k�u@A�(B�v�u?I\Ɏ�@�@*	�Zd;7d@2F
Iterator::ModeljhwH1�?!}7��'�C@)kg{�?1g� X�7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����R�?!GW?�9@)�~��@��?1wcz��4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��×�"�?!�&��Q8@)H�]�۝?15��2@:Preprocessing2U
Iterator::Model::ParallelMapV2�Z��8��?!%y&e/@)�Z��8��?1%y&e/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$����5�?!��h
�qN@)}�;l"3�?1�ы@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����o҄?!��o%@)����o҄?1��o%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceC���|͂?!B�r!�@)C���|͂?1B�r!�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$0��{�?!�f&��=@)q9^��Iy?1��za�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����@����@!����@      ��!       "	�~k�u@�~k�u@!�~k�u@*      ��!       2	�(B�v�u?�(B�v�u?!�(B�v�u?:	\Ɏ�@�@\Ɏ�@�@!\Ɏ�@�@B      ��!       J      ��!       R      ��!       Z      ��!       JGPUb �"
Ugradient_tape/functional_19/conv2d_transpose_36/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�}\B�$�?!�}\B�$�?"c
Ggradient_tape/functional_19/conv2d_transpose_36/conv2d_transpose/Conv2DConv2D�s{�x�?!���pa�?"[
2functional_19/conv2d_transpose_36/conv2d_transposeConv2DBackpropInputD2�9ܬ?!�e?L�?"k
Agradient_tape/functional_19/conv2d_30/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\X$��?!�e����?"
Ugradient_tape/functional_19/conv2d_transpose_35/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter-p^h*��?!�3�0��?"[
2functional_19/conv2d_transpose_37/conv2d_transposeConv2DBackpropInput���P�?!�ԝ2�'�?"
Ugradient_tape/functional_19/conv2d_transpose_37/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�H�-%8�?!��X��.�?"a
@gradient_tape/functional_19/max_pooling2d_18/MaxPool/MaxPoolGradMaxPoolGrad�b;KJ֛?!(T};��?"k
Agradient_tape/functional_19/conv2d_29/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�"xN��?!W�d`��?"V
8gradient_tape/functional_19/conv2d_transpose_36/ReluGradReluGrad�89L��?!�PُV�?Q      Y@Yb�=p&@a�'���W@qw�	n`N@y�;��c5Y?"�	
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�60.7534% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 