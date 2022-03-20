;;; -*- coding:utf-8; mode:lisp -*-

;;; A1A: Binary classification, Dence data, More bigger data
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter a9a-dim 123)
(defparameter a9a-train (read-data "/mnt/data2/datasets/a9a" a9a-dim))
(defparameter a9a-test (read-data "/mnt/data2/datasets/a9a.t" a9a-dim))

(defparameter perceptron-learner (make-perceptron a9a-dim))
(time (loop repeat 1000 do (train perceptron-learner a9a-train)))
(time (test perceptron-learner a9a-test))

;; EXAMPLES> (time (loop repeat 1000 do (train perceptron-learner a9a-train)))
;; Evaluation took:
;;   7.083 seconds of real time
;;   7.009804 seconds of total run time (6.994631 user, 0.015173 system)
;;   98.97% CPU
;;   24,025,434,550 processor cycles
;;   111,574,960 bytes consed
  
;; NIL
;; EXAMPLES> (time (test perceptron-learner a9a-test))
;; Accuracy: 79.72483%, Correct: 12980, Total: 16281
;; Evaluation took:
;;   0.012 seconds of real time
;;   0.012669 seconds of total run time (0.012639 user, 0.000030 system)
;;   108.33% CPU
;;   45,663,009 processor cycles
;;   2,785,072 bytes consed
  
;; 79.72483
;; 12980
;; 16281

;; Accuracy: 79.72483%, Correct: 12980, Total: 16281

;;; struct version
;; Evaluation took:
;;   6.939 seconds of real time
;;   6.943376 seconds of total run time (6.893643 user, 0.049733 system)
;;   [ Run times consist of 0.176 seconds GC time, and 6.768 seconds non-GC time. ]
;;   100.06% CPU
;;   23,539,955,073 processor cycles
;;   1,674,405,248 bytes consed

;; Evaluation took:
;;   5.726 seconds of real time
;;   5.727636 seconds of total run time (5.701425 user, 0.026211 system)
;;   [ Run times consist of 0.059 seconds GC time, and 5.669 seconds non-GC time. ]
;;   100.03% CPU
;;   19,423,247,584 processor cycles
;;   1,674,403,904 bytes consed

;; Evaluation took:
;;   4.841 seconds of real time
;;   4.844107 seconds of total run time (4.837360 user, 0.006747 system)
;;   [ Run times consist of 0.019 seconds GC time, and 4.826 seconds non-GC time. ]
;;   100.06% CPU
;;   16,420,864,642 processor cycles
;;   1,674,411,648 bytes consed

;; use f! and declare in -UPDATE function
;; Evaluation took:
;;   4.037 seconds of real time
;;   4.038199 seconds of total run time (4.027196 user, 0.011003 system)
;;   100.02% CPU
;;   13,694,338,298 processor cycles
;;   111,580,640 bytes consed

;;; CLOS version
;; Evaluation took:
;;   7.647 seconds of real time
;;   7.653533 seconds of total run time (7.653533 user, 0.000000 system)
;;   [ Run times consist of 0.041 seconds GC time, and 7.613 seconds non-GC time. ]
;;   100.09% CPU
;;   25,939,324,287 processor cycles
;;   1,674,404,304 bytes consed

(defparameter a9a-train.sp (read-data "/mnt/data2/datasets/a9a" a9a-dim :sparse-p t))
(defparameter a9a-test.sp (read-data "/mnt/data2/datasets/a9a.t" a9a-dim :sparse-p t))

(defparameter perceptron-learner.sp (make-sparse-perceptron a9a-dim))

;;(sparse-perceptron-update perceptron-learner.sp (cdar a9a-train.sp) (caar a9a-train.sp))

(time (loop repeat 1000 do (train perceptron-learner.sp a9a-train.sp)))
(test perceptron-learner.sp a9a-test.sp)

;;; Sparse version
;; Evaluation took:
;;   1.850 seconds of real time
;;   1.854306 seconds of total run time (1.854306 user, 0.000000 system)
;;   [ Run times consist of 0.024 seconds GC time, and 1.831 seconds non-GC time. ]
;;   100.22% CPU
;;   6,275,038,645 processor cycles
;;   1,674,553,424 bytes consed

;; use sf!, ds-dot!
;; Evaluation took:
;;   1.087 seconds of real time
;;   1.087332 seconds of total run time (1.079737 user, 0.007595 system)
;;   [ Run times consist of 0.004 seconds GC time, and 1.084 seconds non-GC time. ]
;;   100.00% CPU
;;   3,685,724,247 processor cycles
;;   111,641,760 bytes consed

(defparameter arow-learner (make-arow a9a-dim 10.0))
(time (loop repeat 1000 do (train arow-learner a9a-train)))
(test arow-learner a9a-test)

;; Accuracy: 84.964066%, Correct: 13833, Total: 16281

;;; struct version
;; Evaluation took:
;;   27.638 seconds of real time
;;   27.651715 seconds of total run time (27.637809 user, 0.013906 system)
;;   [ Run times consist of 0.143 seconds GC time, and 27.509 seconds non-GC time. ]
;;   100.05% CPU
;;   93,753,921,608 processor cycles
;;   5,881,035,456 bytes consed

;; WARNING: redefining CL-ONLINE-LEARNING.STRUCT:AROW-UPDATE in DEFUN
;; Evaluation took:
;;   26.429 seconds of real time
;;   26.443882 seconds of total run time (26.439552 user, 0.004330 system)
;;   [ Run times consist of 0.154 seconds GC time, and 26.290 seconds non-GC time. ]
;;   100.06% CPU
;;   89,649,174,008 processor cycles
;;   5,881,039,472 bytes consed

;; Evaluation took:
;;   26.151 seconds of real time
;;   26.161610 seconds of total run time (26.161610 user, 0.000000 system)
;;   [ Run times consist of 0.067 seconds GC time, and 26.095 seconds non-GC time. ]
;;   100.04% CPU
;;   88,708,565,423 processor cycles
;;   5,881,035,456 bytes consed

;; Evaluation took:
;;   14.978 seconds of real time
;;   14.987884 seconds of total run time (14.987565 user, 0.000319 system)
;;   [ Run times consist of 0.040 seconds GC time, and 14.948 seconds non-GC time. ]
;;   100.07% CPU
;;   50,808,360,960 processor cycles
;;   5,881,124,352 bytes consed

;; Evaluation took:
;;   13.638 seconds of real time
;;   13.640413 seconds of total run time (13.616424 user, 0.023989 system)
;;   [ Run times consist of 0.066 seconds GC time, and 13.575 seconds non-GC time. ]
;;   100.01% CPU
;;   46,262,175,788 processor cycles
;;   2,307,751,392 bytes consed

;; Evaluation took:
;;   13.452 seconds of real time
;;   13.456763 seconds of total run time (13.454677 user, 0.002086 system)
;;   [ Run times consist of 0.018 seconds GC time, and 13.439 seconds non-GC time. ]
;;   100.04% CPU
;;   45,631,459,380 processor cycles
;;   1,265,773,440 bytes consed

;; Evaluation took:
;;   13.324 seconds of real time
;;   13.329676 seconds of total run time (13.321704 user, 0.007972 system)
;;   [ Run times consist of 0.018 seconds GC time, and 13.312 seconds non-GC time. ]
;;   100.05% CPU
;;   45,197,249,772 processor cycles
;;   1,265,792,672 bytes consed

;;; CLOS version
;; Evaluation took:
;;   31.187 seconds of real time
;;   31.198607 seconds of total run time (31.189536 user, 0.009071 system)
;;   [ Run times consist of 0.140 seconds GC time, and 31.059 seconds non-GC time. ]
;;   100.04% CPU
;;   105,792,514,974 processor cycles
;;   5,881,026,464 bytes consed

(defparameter arow-learner.sp (make-sparse-arow a9a-dim 10.0))
(time (loop repeat 1000 do (train arow-learner.sp a9a-train.sp)))
(test arow-learner.sp a9a-test.sp)

;; Evaluation took:
;;   5.300 seconds of real time
;;   5.304542 seconds of total run time (5.289788 user, 0.014754 system)
;;   [ Run times consist of 0.081 seconds GC time, and 5.224 seconds non-GC time. ]
;;   100.09% CPU
;;   17,976,932,767 processor cycles
;;   5,881,157,952 bytes consed

;; use sf!, ds-dot!
;; Evaluation took:
;;   4.636 seconds of real time
;;   4.644716 seconds of total run time (4.644704 user, 0.000012 system)
;;   [ Run times consist of 0.074 seconds GC time, and 4.571 seconds non-GC time. ]
;;   100.19% CPU
;;   15,726,641,730 processor cycles
;;   4,113,667,680 bytes consed

;; declare type
;; Evaluation took:
;;   3.473 seconds of real time
;;   3.478154 seconds of total run time (3.474138 user, 0.004016 system)
;;   [ Run times consist of 0.041 seconds GC time, and 3.438 seconds non-GC time. ]
;;   100.14% CPU
;;   11,781,892,039 processor cycles
;;   2,215,116,144 bytes consed

;; Evaluation took:
;;   3.175 seconds of real time
;;   3.176213 seconds of total run time (3.176213 user, 0.000000 system)
;;   [ Run times consist of 0.027 seconds GC time, and 3.150 seconds non-GC time. ]
;;   100.03% CPU
;;   10,767,676,763 processor cycles
;;   1,265,828,608 bytes consed

(defparameter scw-learner (make-scw a9a-dim 0.9 0.1))
(time (loop repeat 1000 do (train scw-learner a9a-train)))
(test scw-learner a9a-test)

;; Accuracy: 83.98133%, Correct: 13673, Total: 16281

;; Evaluation took:
;;   13.235 seconds of real time
;;   13.247439 seconds of total run time (13.239742 user, 0.007697 system)
;;   [ Run times consist of 0.093 seconds GC time, and 13.155 seconds non-GC time. ]
;;   100.09% CPU
;;   44,896,182,054 processor cycles
;;   7,375,026,592 bytes consed

;; Evaluation took:
;;   11.180 seconds of real time
;;   11.142296 seconds of total run time (11.133413 user, 0.008883 system)
;;   [ Run times consist of 0.022 seconds GC time, and 11.121 seconds non-GC time. ]
;;   99.66% CPU
;;   37,925,366,659 processor cycles
;;   1,644,783,376 bytes consed

;; Evaluation took:
;;   10.753 seconds of real time
;;   10.702306 seconds of total run time (10.690146 user, 0.012160 system)
;;   [ Run times consist of 0.005 seconds GC time, and 10.698 seconds non-GC time. ]
;;   99.53% CPU
;;   36,474,629,265 processor cycles
;;   329,023,488 bytes consed

;; Evaluation took:
;;   10.509 seconds of real time
;;   10.511938 seconds of total run time (10.500368 user, 0.011570 system)
;;   [ Run times consist of 0.005 seconds GC time, and 10.507 seconds non-GC time. ]
;;   100.03% CPU
;;   35,648,299,084 processor cycles
;;   329,023,360 bytes consed

(defparameter scw-learner.sp (make-sparse-scw a9a-dim 0.9 0.1))
(time (loop repeat 1000 do (train scw-learner.sp a9a-train.sp)))
(test scw-learner.sp a9a-test.sp)

;; Evaluation took:
;;   5.443 seconds of real time
;;   5.450985 seconds of total run time (5.439000 user, 0.011985 system)
;;   [ Run times consist of 0.105 seconds GC time, and 5.346 seconds non-GC time. ]
;;   100.15% CPU
;;   18,461,632,884 processor cycles
;;   7,375,126,240 bytes consed

;; Evaluation took:
;;   2.524 seconds of real time
;;   2.525696 seconds of total run time (2.525696 user, 0.000000 system)
;;   [ Run times consist of 0.009 seconds GC time, and 2.517 seconds non-GC time. ]
;;   100.08% CPU
;;   8,564,202,770 processor cycles
;;   329,047,584 bytes consed

(defparameter lr+sgd-learner (make-lr+sgd a9a-dim 0.001 0.001))
(time (loop repeat 1000 do (train lr+sgd-learner a9a-train)))
(test lr+sgd-learner a9a-test)

;; Evaluation took:
;;   15.391 seconds of real time
;;   15.395063 seconds of total run time (15.391215 user, 0.003848 system)
;;   [ Run times consist of 0.064 seconds GC time, and 15.332 seconds non-GC time. ]
;;   100.03% CPU
;;   52,207,640,106 processor cycles
;;   3,646,916,864 bytes consed

;; use logistic-regression-gradient!
;; Evaluation took:
;;   14.888 seconds of real time
;;   14.891887 seconds of total run time (14.883481 user, 0.008406 system)
;;   [ Run times consist of 0.026 seconds GC time, and 14.866 seconds non-GC time. ]
;;   100.03% CPU
;;   50,502,032,097 processor cycles
;;   1,562,999,200 bytes consed
  
;; Accuracy: 85.129906%, Correct: 13860, Total: 16281

(defparameter lr+sgd-learner.sp (make-sparse-lr+sgd a9a-dim 0.001 0.001))
(time (loop repeat 1000 do (train lr+sgd-learner.sp a9a-train.sp)))
(test lr+sgd-learner.sp a9a-test.sp)

;; Evaluation took:
;;   8.391 seconds of real time
;;   8.391694 seconds of total run time (8.378601 user, 0.013093 system)
;;   [ Run times consist of 0.026 seconds GC time, and 8.366 seconds non-GC time. ]
;;   100.01% CPU
;;   28,462,239,673 processor cycles
;;   1,563,021,344 bytes consed

;; Accuracy: 85.129906%, Correct: 13860, Total: 16281

(defparameter adam-learner (make-lr+adam a9a-dim 0.001 0.001 1.e-8 0.9 0.99))
(time (loop repeat 1000 do (train adam-learner a9a-train)))
(test  adam-learner a9a-test)

;; Evaluation took:
;;   67.682 seconds of real time
;;   67.706814 seconds of total run time (67.671638 user, 0.035176 system)
;;   [ Run times consist of 0.197 seconds GC time, and 67.510 seconds non-GC time. ]
;;   100.04% CPU
;;   229,585,538,903 processor cycles
;;   15,108,377,152 bytes consed

;; Accuracy: 84.957924%, Correct: 13832, Total: 16281

;; Evaluation took:
;;   62.894 seconds of real time
;;   63.634589 seconds of total run time (63.634589 user, 0.000000 system)
;;   101.18% CPU
;;   213,348,017,096 processor cycles
;;   6,772,740,016 bytes consed

(defparameter adam-learner.sp (make-sparse-lr+adam a9a-dim 0.001 0.001 1.e-8 0.9 0.99))
(time (loop repeat 1000 do (train adam-learner.sp a9a-train.sp)))
(test  adam-learner.sp a9a-test.sp)

;; Evaluation took:
;;   57.980 seconds of real time
;;   58.001587 seconds of total run time (57.951849 user, 0.049738 system)
;;   [ Run times consist of 0.116 seconds GC time, and 57.886 seconds non-GC time. ]
;;   100.04% CPU
;;   196,675,200,604 processor cycles
;;   6,772,815,744 bytes consed

;; Accuracy: 84.957924%, Correct: 13832, Total: 16281

;;; AROW++
;; ~/datasets $ arow_learn -i 1000 a9a a9a.arow.model
;; Number of features: 123
;; Number of examples: 32561
;; Number of updates:  19757376
;; Done!
;; Time: 58.0621 sec.

(require :sb-sprof)

(sb-sprof:with-profiling (:max-samples 10000
				       :mode :alloc
				       :report :flat)
  (loop repeat 100 do (train adam-learner a9a-train)))
