  *	???K7?l@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate)=?K?e??!??]ҶO@)?YKi???1NM?5NaN@:Preprocessing2T
Iterator::Root::ParallelMapV2?Z?a/??!?????R4@)?Z?a/??1?????R4@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???5?e??!?i?? _ @)QL? 3??1??CU@:Preprocessing2E
Iterator::RootH,????!?????8@)??b('څ?1????vq@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zips?m?B<??!9?YD?R@)????C?1 ?=70c
@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorWzm6Vbn?!???????)Wzm6Vbn?1???????:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap@?? k???!?LJ???O@)?E'K??k?1v???Ú??:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor
-???]?!A???????)
-???]?1A???????:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceN?E? V?!EӣQ5???)N?E? V?1EӣQ5???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.