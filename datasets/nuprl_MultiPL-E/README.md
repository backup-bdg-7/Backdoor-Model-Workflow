---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
- expert-generated
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
- extended|openai_humaneval
- extended|mbpp
task_categories: []
task_ids: []
pretty_name: MultiPLE-E
tags: []
dataset_info:
- config_name: humaneval-adb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 259548
    num_examples: 157
  download_size: 76995
  dataset_size: 259548
- config_name: humaneval-clj
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 174890
    num_examples: 161
  download_size: 70395
  dataset_size: 174890
- config_name: humaneval-cpp
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 245061
    num_examples: 161
  download_size: 83221
  dataset_size: 245061
- config_name: humaneval-cs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 288571
    num_examples: 158
  download_size: 82080
  dataset_size: 288571
- config_name: humaneval-d
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 179391
    num_examples: 156
  download_size: 70027
  dataset_size: 179391
- config_name: humaneval-dart
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 240233
    num_examples: 157
  download_size: 75805
  dataset_size: 240233
- config_name: humaneval-elixir
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 207052
    num_examples: 161
  download_size: 74798
  dataset_size: 207052
- config_name: humaneval-go
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 252128
    num_examples: 154
  download_size: 78121
  dataset_size: 252128
- config_name: humaneval-hs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 210523
    num_examples: 156
  download_size: 69373
  dataset_size: 210523
- config_name: humaneval-java
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 293293
    num_examples: 158
  download_size: 86178
  dataset_size: 293293
- config_name: humaneval-jl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 165943
    num_examples: 159
  download_size: 68620
  dataset_size: 165943
- config_name: humaneval-js
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 187162
    num_examples: 161
  download_size: 70034
  dataset_size: 187162
- config_name: humaneval-lua
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 190211
    num_examples: 161
  download_size: 70547
  dataset_size: 190211
- config_name: humaneval-ml
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 169037
    num_examples: 155
  download_size: 68199
  dataset_size: 169037
- config_name: humaneval-php
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 230721
    num_examples: 161
  download_size: 75195
  dataset_size: 230721
- config_name: humaneval-pl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 248652
    num_examples: 161
  download_size: 77247
  dataset_size: 248652
- config_name: humaneval-r
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 195050
    num_examples: 161
  download_size: 71602
  dataset_size: 195050
- config_name: humaneval-rb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 193448
    num_examples: 161
  download_size: 72942
  dataset_size: 193448
- config_name: humaneval-rkt
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 194898
    num_examples: 161
  download_size: 70785
  dataset_size: 194898
- config_name: humaneval-rs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 193677
    num_examples: 156
  download_size: 75300
  dataset_size: 193677
- config_name: humaneval-scala
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 245564
    num_examples: 160
  download_size: 80950
  dataset_size: 245564
- config_name: humaneval-sh
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 169419
    num_examples: 158
  download_size: 67691
  dataset_size: 169419
- config_name: humaneval-swift
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 209818
    num_examples: 158
  download_size: 78057
  dataset_size: 209818
- config_name: humaneval-ts
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 187330
    num_examples: 159
  download_size: 70294
  dataset_size: 187330
- config_name: mbpp-adb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 417220
    num_examples: 365
  download_size: 100314
  dataset_size: 417220
- config_name: mbpp-clj
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 249203
    num_examples: 397
  download_size: 76741
  dataset_size: 249203
- config_name: mbpp-cpp
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 362938
    num_examples: 397
  download_size: 97734
  dataset_size: 362938
- config_name: mbpp-cs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 418542
    num_examples: 386
  download_size: 99239
  dataset_size: 418542
- config_name: mbpp-d
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 233997
    num_examples: 358
  download_size: 73269
  dataset_size: 233997
- config_name: mbpp-elixir
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 299264
    num_examples: 397
  download_size: 84803
  dataset_size: 299264
- config_name: mbpp-go
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 401215
    num_examples: 374
  download_size: 93635
  dataset_size: 401215
- config_name: mbpp-hs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 256021
    num_examples: 355
  download_size: 71870
  dataset_size: 256021
- config_name: mbpp-java
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 424038
    num_examples: 386
  download_size: 99991
  dataset_size: 424038
- config_name: mbpp-jl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 229892
    num_examples: 390
  download_size: 77046
  dataset_size: 229892
- config_name: mbpp-js
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 259131
    num_examples: 397
  download_size: 78109
  dataset_size: 259131
- config_name: mbpp-lua
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 265029
    num_examples: 397
  download_size: 78701
  dataset_size: 265029
- config_name: mbpp-ml
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 208995
    num_examples: 355
  download_size: 69995
  dataset_size: 208995
- config_name: mbpp-php
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 311660
    num_examples: 397
  download_size: 82614
  dataset_size: 311660
- config_name: mbpp-pl
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 323620
    num_examples: 396
  download_size: 83295
  dataset_size: 323620
- config_name: mbpp-r
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 259911
    num_examples: 397
  download_size: 78685
  dataset_size: 259911
- config_name: mbpp-rb
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 269278
    num_examples: 397
  download_size: 82986
  dataset_size: 269278
- config_name: mbpp-rkt
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 271330
    num_examples: 397
  download_size: 77882
  dataset_size: 271330
- config_name: mbpp-rs
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 220467
    num_examples: 354
  download_size: 72084
  dataset_size: 220467
- config_name: mbpp-scala
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 333175
    num_examples: 396
  download_size: 92626
  dataset_size: 333175
- config_name: mbpp-sh
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 219417
    num_examples: 382
  download_size: 69685
  dataset_size: 219417
- config_name: mbpp-swift
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 320342
    num_examples: 396
  download_size: 89609
  dataset_size: 320342
- config_name: mbpp-ts
  features:
  - name: name
    dtype: string
  - name: language
    dtype: string
  - name: prompt
    dtype: string
  - name: doctests
    dtype: string
  - name: original
    dtype: string
  - name: prompt_terminology
    dtype: string
  - name: tests
    dtype: string
  - name: stop_tokens
    sequence: string
  splits:
  - name: test
    num_bytes: 268597
    num_examples: 390
  download_size: 78505
  dataset_size: 268597
configs:
- config_name: humaneval-adb
  data_files:
  - split: test
    path: humaneval-adb/test-*
- config_name: humaneval-clj
  data_files:
  - split: test
    path: humaneval-clj/test-*
- config_name: humaneval-cpp
  data_files:
  - split: test
    path: humaneval-cpp/test-*
- config_name: humaneval-cs
  data_files:
  - split: test
    path: humaneval-cs/test-*
- config_name: humaneval-d
  data_files:
  - split: test
    path: humaneval-d/test-*
- config_name: humaneval-dart
  data_files:
  - split: test
    path: humaneval-dart/test-*
- config_name: humaneval-elixir
  data_files:
  - split: test
    path: humaneval-elixir/test-*
- config_name: humaneval-go
  data_files:
  - split: test
    path: humaneval-go/test-*
- config_name: humaneval-hs
  data_files:
  - split: test
    path: humaneval-hs/test-*
- config_name: humaneval-java
  data_files:
  - split: test
    path: humaneval-java/test-*
- config_name: humaneval-jl
  data_files:
  - split: test
    path: humaneval-jl/test-*
- config_name: humaneval-js
  data_files:
  - split: test
    path: humaneval-js/test-*
- config_name: humaneval-lua
  data_files:
  - split: test
    path: humaneval-lua/test-*
- config_name: humaneval-ml
  data_files:
  - split: test
    path: humaneval-ml/test-*
- config_name: humaneval-php
  data_files:
  - split: test
    path: humaneval-php/test-*
- config_name: humaneval-pl
  data_files:
  - split: test
    path: humaneval-pl/test-*
- config_name: humaneval-r
  data_files:
  - split: test
    path: humaneval-r/test-*
- config_name: humaneval-rb
  data_files:
  - split: test
    path: humaneval-rb/test-*
- config_name: humaneval-rkt
  data_files:
  - split: test
    path: humaneval-rkt/test-*
- config_name: humaneval-rs
  data_files:
  - split: test
    path: humaneval-rs/test-*
- config_name: humaneval-scala
  data_files:
  - split: test
    path: humaneval-scala/test-*
- config_name: humaneval-sh
  data_files:
  - split: test
    path: humaneval-sh/test-*
- config_name: humaneval-swift
  data_files:
  - split: test
    path: humaneval-swift/test-*
- config_name: humaneval-ts
  data_files:
  - split: test
    path: humaneval-ts/test-*
- config_name: mbpp-adb
  data_files:
  - split: test
    path: mbpp-adb/test-*
- config_name: mbpp-clj
  data_files:
  - split: test
    path: mbpp-clj/test-*
- config_name: mbpp-cpp
  data_files:
  - split: test
    path: mbpp-cpp/test-*
- config_name: mbpp-cs
  data_files:
  - split: test
    path: mbpp-cs/test-*
- config_name: mbpp-d
  data_files:
  - split: test
    path: mbpp-d/test-*
- config_name: mbpp-elixir
  data_files:
  - split: test
    path: mbpp-elixir/test-*
- config_name: mbpp-go
  data_files:
  - split: test
    path: mbpp-go/test-*
- config_name: mbpp-hs
  data_files:
  - split: test
    path: mbpp-hs/test-*
- config_name: mbpp-java
  data_files:
  - split: test
    path: mbpp-java/test-*
- config_name: mbpp-jl
  data_files:
  - split: test
    path: mbpp-jl/test-*
- config_name: mbpp-js
  data_files:
  - split: test
    path: mbpp-js/test-*
- config_name: mbpp-lua
  data_files:
  - split: test
    path: mbpp-lua/test-*
- config_name: mbpp-ml
  data_files:
  - split: test
    path: mbpp-ml/test-*
- config_name: mbpp-php
  data_files:
  - split: test
    path: mbpp-php/test-*
- config_name: mbpp-pl
  data_files:
  - split: test
    path: mbpp-pl/test-*
- config_name: mbpp-r
  data_files:
  - split: test
    path: mbpp-r/test-*
- config_name: mbpp-rb
  data_files:
  - split: test
    path: mbpp-rb/test-*
- config_name: mbpp-rkt
  data_files:
  - split: test
    path: mbpp-rkt/test-*
- config_name: mbpp-rs
  data_files:
  - split: test
    path: mbpp-rs/test-*
- config_name: mbpp-scala
  data_files:
  - split: test
    path: mbpp-scala/test-*
- config_name: mbpp-sh
  data_files:
  - split: test
    path: mbpp-sh/test-*
- config_name: mbpp-swift
  data_files:
  - split: test
    path: mbpp-swift/test-*
- config_name: mbpp-ts
  data_files:
  - split: test
    path: mbpp-ts/test-*
---

# Dataset Card for MultiPL-E

## Dataset Description

- **Repository:**  https://github.com/nuprl/MultiPL-E
- **Paper:** https://ieeexplore.ieee.org/abstract/document/10103177
- **Point of Contact:** carolyn.anderson@wellesley.edu, mfeldman@oberlin.edu, a.guha@northeastern.edu

## Dataset Summary

MultiPL-E is a dataset for evaluating large language models for code
generation that supports 22 programming languages. It takes the OpenAI 
HumanEval and the Mostly Basic Python Programs (MBPP) benchmarks and uses little compilers to
translate them  to other languages. It is easy to add support for new languages 
and benchmarks.

The dataset is divided into several configurations named *SRCDATA-LANG*, where
*SRCDATA* is either "humaneval" or "mbpp" and *LANG* is one of the supported
languages. We use the canonical file extension for each language to identify
the language, e.g., "cpp" for C++, "lua" for Lua, "clj" for Clojure, and so on.

## Using MultiPL-E

- MultiPL-E is part of the [BigCode Code Generation LM Harness]. This
  is the easiest way to use MultiPL-E.

- MultiPL-E has its own evaluation framework that supports proprietary models,
  the prompt ablations, more source benchmarks, and more recently added
  programming languages. See the [MultiPL-E tutorial] on how to use this
  framework directly.

## The MultiPL-E Ablations

The MultiPL-E paper presented several ablations of the prompt for the original
set of programming languages. We do not include them in the current version of
MultiPL-E, but they are still available in this repository from revision
`d23b094` or earlier. (You can optionally pass the revision to
`datasets.load_dataset`.)

These are the prompt variations:

- *SRCDATA-LANG-keep* is the same as *SRCDATA-LANG*, but the text of the prompt
  is totally unchanged. If the original prompt had Python doctests, they remain
  as Python instead of being translated to *LANG*. If the original prompt had 
  Python-specific terminology, e.g., "list", it remains "list", instead of 
  being translated, e.g., to "vector" for C++.

- *SRCDATA-LANG-transform* transforms the doctests to *LANG* but leaves
  the natural language text of the prompt unchanged.

- *SRCDATA-LANG-removed* removes the doctests from the prompt.

Note that MBPP does not have any doctests, so the "removed" and "transform"
variations are not available for MBPP.

## Changelog

### Version 3.2

MultiPL-E now supports Ada, thanks to [Rowan Walshe](https://github.com/rowan-walshe).
Rowan identified some issues that likely have a small negative impact on the benchmark
scores for existing languages. We have not updated the prompts for those languages
at this time. See the discussions [PR 162](https://github.com/nuprl/MultiPL-E/pull/162)
and [PR 163](https://github.com/nuprl/MultiPL-E/pull/163).


### Version 3.1.1

This version fixes a bug that affected some TypeScript problems, thanks to [Niels Mündler
](https://github.com/nielstron). The issue impacts MBPP-based problems. The fix changes
whitespace in a few HumanEval-based problems that should be insignificant. These
are the relevant changes:

```diff
=== mbpp-ts_prompt_mbpp_253_count_integer.diff ===
- function count_integer(list1: number| string| number[]): number {
+ function count_integer(list1: (number | string | number)[]): number {
=== mbpp-ts_prompt_mbpp_278_count_first_elements.diff ===
- function count_first_elements(test_tup: number| [number, number][]): number {
+ function count_first_elements(test_tup: (number | [number, number])[]): number {
=== mbpp-ts_prompt_mbpp_294_max_val.diff ===
- function max_val(listval: string| number[]): number {
+ function max_val(listval: (string | number)[]): number {
=== mbpp-ts_prompt_mbpp_297_flatten_list.diff ===
- function flatten_list(list1: number| number[][]): number[] {
+ function flatten_list(list1: (number | number[])[]): number[] {
=== mbpp-ts_prompt_mbpp_405_check_tuplex.diff ===
- function check_tuplex(tuplex: string| number[], tuple1: any): boolean {
+ function check_tuplex(tuplex: (string | number)[], tuple1: any): boolean {
=== mbpp-ts_prompt_mbpp_410_min_val.diff ===
- function min_val(listval: string| number[]): number {
+ function min_val(listval: (string | number)[]): number {
=== mbpp-ts_prompt_mbpp_419_round_and_sum.diff ===
- function round_and_sum(list1: number| number[]): number {
+ function round_and_sum(list1: (number | number)[]): number {
=== mbpp-ts_prompt_mbpp_65_recursive_list_sum.diff ===
- function recursive_list_sum(data_list: number| number[][]): number {
+ function recursive_list_sum(data_list: (number | number[])[]): number {
=== mbpp-ts_prompt_mbpp_755_second_smallest.diff ===
- function second_smallest(numbers: number| number[]): number | undefined {
+ function second_smallest(numbers: (number | number)[]): number | undefined {
```

See [Github Issue 160](https://github.com/nuprl/MultiPL-E/issues/160) for more
information.

### Version 3.1

MultiPL-E now supports Dart, thanks to [Devon Carew](https://github.com/devoncarew).

### Version 3.0

This is the first significant update since MultiPL-E was used in StarCoder 1.

1. The dataset was versioned at 3.0, and we are bumping the software version to stay in sync.
2. We no longer publish the MultiPL-E ablations, but they are available in
   revision `d23b094` and earlier.
3. New programming languages supported:
   - Clojure, thanks to [Alex Miller](https://github.com/puredanger)
   - Elixir, thanks to [Marko Vukovic](https://github.com/mvkvc)
   - Haskell, thanks to [Thomas Dwyer](https://github.com/Cajunvoodoo)
   - OCaml, thanks to [John Gouwar](https://johngouwar.github.io)
4. Changes to existing HumanEval-based problems:
   - Four Scala problems have fixed prompts/tests (12, 90, 128, 162).
   - Some whitespace-only changes to problems for Racket (18 problems),
     R (36 problems), Julia (159 problems), and D (156 problems). We will try to
     avoid these kinds of changes in the future.
5. The MBPP-based problems have changes analogous to the HumanEval-based problems.
   
See the directory `diffs_v3.0` in the dataset repository for the diffs to
each prompt.

### Version 0.5.0

Instruction-following support and new languages

  - New languages: Luau, Elixir, Lean, Coq, Dafny
  - Support for instruction-following prompts
  - vLLM support for faster evaluation

### Version 0.4.0

QoL improvements and new languages

  - New languages: OCaml, MATLAB
  - Using `.jsonl` instead of `.json` for prompts
  - Several bugfixes to prompts

### Version 0.3.0

- This version was used to evaluate [StarCoder]

- This version corrects several bugs in prompts and test cases that resulted in lower
  pass@k rates for some of the statically typed languages. The most significant difference
  is that the pass@k for Java increases by about 2% on HumanEval.

### Version 0.2.0

This version was used to evaluate [SantaCoder]

[SantaCoder]: https://arxiv.org/abs/2301.03988
[StarCoder]: https://arxiv.org/abs/2305.06161
[BigCode Code Generation LM Harness]: https://github.com/bigcode-project/bigcode-evaluation-harness
[MultiPL-E tutorial]: https://nuprl.github.io/MultiPL-E/