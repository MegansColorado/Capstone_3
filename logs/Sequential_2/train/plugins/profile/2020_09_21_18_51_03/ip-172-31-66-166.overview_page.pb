�	&jjٙk@&jjٙk@!&jjٙk@	����q��?����q��?!����q��?"n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-&jjٙk@q8�9@�?1w�Nyt5k@I�Ss����?Y�,'���?*	h��|?E^@2F
Iterator::Model�?�&M��?!��K)bE@)���ޞ?1D�+&A�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatӿ$�)�?!����[B;@)��r���?11�
�6[3@:Preprocessing2U
Iterator::Model::ParallelMapV2��E��(�?!iVq�1@)��E��(�?1iVq�1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���-��?!��E42@)��x!�?1_���I(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?�J���?!\�>�֝L@)-��VЄ?1{vQ�v� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor#�qp阃?!���@)#�qp阃?1���@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��I`s~?!K���=@)��I`s~?1K���=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�%ǝ���?!�I.��5@)���$��p?1*��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9����q��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	q8�9@�?q8�9@�?!q8�9@�?      ��!       "	w�Nyt5k@w�Nyt5k@!w�Nyt5k@*      ��!       2      ��!       :	�Ss����?�Ss����?!�Ss����?B      ��!       J	�,'���?�,'���?!�,'���?R      ��!       Z	�,'���?�,'���?!�,'���?JGPUY����q��?b �"}
Sgradient_tape/sequential_1/conv2d_transpose_8/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter��J�4��?!��J�4��?"a
Egradient_tape/sequential_1/conv2d_transpose_8/conv2d_transpose/Conv2DConv2D<�l�0��?!��[�2��?"Y
0sequential_1/conv2d_transpose_8/conv2d_transposeConv2DBackpropInputD&�z�C�?!s��(,�?"Y
0sequential_1/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput��$�?!4������?"g
>gradient_tape/sequential_1/conv2d_8/Conv2D/Conv2DBackpropInputConv2DBackpropInput_p%�?!pۚ�W�?"}
Sgradient_tape/sequential_1/conv2d_transpose_9/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter�}����?!KS��5�?"~
Tgradient_tape/sequential_1/conv2d_transpose_10/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter1C��.��?!~�ԀL�?"i
?gradient_tape/sequential_1/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter� Z�7�?!�I�����?"i
?gradient_tape/sequential_1/conv2d_9/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter�(�/2�?!�ע���?"I
+gradient_tape/sequential_1/dense_5/ReluGradReluGradbx�xr�?!���.~�?Q      Y@Y��8��8&@a��8��8V@qO#>�8@y�@V�IY?"�	
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQb�24.9853% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 