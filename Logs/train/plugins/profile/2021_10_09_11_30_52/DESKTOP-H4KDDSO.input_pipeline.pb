  *	
ףp=?R@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate2???3??!K|]??uC@)?V?????11.B@:Preprocessing2T
Iterator::Root::ParallelMapV2??C????!:_R?d?1@)??C????1:_R?d?1@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatY??Z???!???:<
6@)??p?Qe??1FV\?o/@:Preprocessing2E
Iterator::Root???ȭI??!?:l2>@)EJ?y??1??3"??(@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?>??s(??!A??ysQ@)!;oc?#u?1ն??~=@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?
E??s?!з)2?I@)?
E??s?1з)2?I@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap,?IEc???!,????D@)?/K;5?[?1n????@:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorc???JU?!Z??O??)c???JU?1Z??O??:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?]?pXJ?!x??	n???)?]?pXJ?1x??	n???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.