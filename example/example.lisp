;;; -*- coding:utf-8; mode:lisp -*-

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;; Examples ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

;;; Binary classification, Dence data

(defparameter a1a-dim 123)
(defparameter a1a-train (read-libsvm-data "/home/wiz/datasets/a1a" a1a-dim))
(defparameter a1a-test (read-libsvm-data "/home/wiz/datasets/a1a.t" a1a-dim))

(defparameter perceptron-learner (make-perceptron a1a-dim))
(train perceptron-learner a1a-train)
(test perceptron-learner a1a-test)

(defparameter arow-learner (make-arow a1a-dim 10d0))
(train arow-learner a1a-train)
(test arow-learner a1a-test)

(defparameter scw-learner (make-scw  a1a-dim 0.9d0 0.1d0))
(train scw-learner a1a-train)
(test scw-learner a1a-test)

(defparameter sgd-learner (clol::make-sgd a1a-dim 0.00001d0 0.01d0))
(train sgd-learner a1a-train)
(test sgd-learner a1a-test)

(loop for C in '(0d0 0.00001d0 0.0001d0 0.001d0 0.01d0 0.1d0 1d0) do
  (defparameter sgd-learner (clol::make-sgd a1a-dim C 0.01d0))
  (print (clol::sgd-C sgd-learner))
  (loop repeat 20 do
    (train sgd-learner a1a-train)
    (test  sgd-learner a1a-test)))

; α = 0.001, β1 = 0.9, β2 = 0.999 and ε = 10^-8
(defparameter adam-learner (make-adam a1a-dim 0.000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(adam-update adam-learner (cdar a1a-train) (caar a1a-train))

(ql:quickload :clgplot)

(clgp:plot (reverse clol::*dbg*))

(loop repeat 10 do
  (adam-train adam-learner a1a-train)
  (adam-test adam-learner a1a-test))

(loop for C in '(0d0 0.00001d0 0.0001d0 0.001d0 0.01d0 0.1d0 1d0) do
  (defparameter adam-learner (make-adam a1a-dim C 0.001d0 1.d-8 0.9d0 0.99d0))
  (print (clol::adam-C adam-learner))
  (loop repeat 20 do
    (train adam-learner a1a-train)
    (test  adam-learner a1a-test)))

;;; Binary classification, Sparse data

(defparameter sparse-a1a-train (read-libsvm-data-sparse "/home/wiz/tmp/a1a"))
(defparameter sparse-a1a-test (read-libsvm-data-sparse "/home/wiz/tmp/a1a.t"))

(defparameter sparse-perceptron-learner (make-sparse-perceptron a1a-dim))
(train sparse-perceptron-learner sparse-a1a-train)
(test sparse-perceptron-learner sparse-a1a-test)

(defparameter sparse-arow-learner (make-sparse-arow a1a-dim 10d0))
(train sparse-arow-learner sparse-a1a-train)
(test sparse-arow-learner sparse-a1a-test)

(defparameter sparse-scw-learner (make-sparse-scw a1a-dim 0.9d0 0.1d0))
(train sparse-scw-learner sparse-a1a-train)
(sparse-scw-test sparse-scw-learner sparse-a1a-test)

;;; More bigger data

(defparameter a9a-dim 123)

(defparameter a9a-train (read-libsvm-data "/home/wiz/tmp/a9a" a9a-dim))
(defparameter a9a-test (read-libsvm-data "/home/wiz/tmp/a9a.t" a9a-dim))

(defparameter perceptron-learner (make-perceptron a9a-dim))
(time (loop repeat 1000 do (train perceptron-learner a9a-train)))
(test perceptron-learner a9a-test)

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

;;; CLOS version
;; Evaluation took:
;;   7.647 seconds of real time
;;   7.653533 seconds of total run time (7.653533 user, 0.000000 system)
;;   [ Run times consist of 0.041 seconds GC time, and 7.613 seconds non-GC time. ]
;;   100.09% CPU
;;   25,939,324,287 processor cycles
;;   1,674,404,304 bytes consed

(defparameter a9a-train.sp (read-libsvm-data-sparse "/home/wiz/tmp/a9a"))
(defparameter a9a-test.sp  (read-libsvm-data-sparse "/home/wiz/tmp/a9a.t"))

(defparameter perceptron-learner.sp (make-sparse-perceptron a9a-dim))
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

(defparameter arow-learner (make-arow a9a-dim 10d0))
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

;;; CLOS version
;; Evaluation took:
;;   31.187 seconds of real time
;;   31.198607 seconds of total run time (31.189536 user, 0.009071 system)
;;   [ Run times consist of 0.140 seconds GC time, and 31.059 seconds non-GC time. ]
;;   100.04% CPU
;;   105,792,514,974 processor cycles
;;   5,881,026,464 bytes consed

(defparameter arow-learner.sp (make-sparse-arow a9a-dim 10d0))
(time (loop repeat 1000 do (train arow-learner.sp a9a-train.sp)))
(test arow-learner.sp a9a-test.sp)

;; Evaluation took:
;;   5.300 seconds of real time
;;   5.304542 seconds of total run time (5.289788 user, 0.014754 system)
;;   [ Run times consist of 0.081 seconds GC time, and 5.224 seconds non-GC time. ]
;;   100.09% CPU
;;   17,976,932,767 processor cycles
;;   5,881,157,952 bytes consed

(defparameter scw-learner (make-scw a9a-dim 0.9d0 0.1d0))
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

(defparameter scw-learner.sp (make-sparse-scw a9a-dim 0.9d0 0.1d0))
(time (loop repeat 1000 do (train scw-learner.sp a9a-train.sp)))
(test scw-learner.sp a9a-test.sp)

;; Evaluation took:
;;   5.443 seconds of real time
;;   5.450985 seconds of total run time (5.439000 user, 0.011985 system)
;;   [ Run times consist of 0.105 seconds GC time, and 5.346 seconds non-GC time. ]
;;   100.15% CPU
;;   18,461,632,884 processor cycles
;;   7,375,126,240 bytes consed

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
  (loop repeat 1000 do (arow-train arow-learner a9a-train)))

;;; Multiclass

(defparameter iris-dim 4)

(defparameter iris
  (shuffle-vector
   (coerce (read-libsvm-data-multiclass "/home/wiz/tmp/iris.scale" iris-dim)
	   'simple-vector)))

(defparameter iris
  (read-libsvm-data-multiclass "/home/wiz/tmp/iris.scale" iris-dim))

(defparameter iris-train (subseq iris 0 100))
(defparameter iris-test (subseq iris 100))

(defparameter mul-percep (make-one-vs-rest 4 3 'perceptron))
(train mul-percep iris)
(test mul-percep iris)

;; AROW
(defparameter mul-arow (make-one-vs-rest 4 3 'arow 0.1d0))
(train mul-arow iris-train)
(test mul-arow iris-test)

(defparameter mul-arow (make-one-vs-one 4 3 'arow 0.1d0))
(train mul-arow iris-train)
(test mul-arow iris-test)

;;;;; MNIST ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter mnist-dim 780)
(defparameter mnist-train (read-libsvm-data-multiclass "/home/wiz/tmp/mnist.scale" mnist-dim))
(defparameter mnist-test  (read-libsvm-data-multiclass "/home/wiz/tmp/mnist.scale.t" mnist-dim))

;; Add 1 to labels because the labels of this dataset begin from 0
(dolist (datum mnist-train) (incf (car datum)))
(dolist (datum mnist-test)  (incf (car datum)))

(defparameter mnist-arow (make-one-vs-one mnist-dim 10 'arow 10d0))
(time (loop repeat 8 do (train mnist-arow mnist-train)))

;; Evaluation took:
;;   3.946 seconds of real time
;;   3.956962 seconds of total run time (3.956962 user, 0.000000 system)
;;   100.28% CPU
;;   13,384,797,419 processor cycles
;;   337,643,712 bytes consed

(test mnist-arow mnist-test)
;; Accuracy: 94.6%, Correct: 9460, Total: 10000

;;; LIBLINEAR
;; wiz@prime:~/tmp$ time liblinear-train -q mnist.scale mnist.model
;; real    2m26.804s
;; user    2m26.668s
;; sys     0m0.312s

;; wiz@prime:~/tmp$ liblinear-predict mnist.scale.t mnist.model mnist.out
;; Accuracy = 91.69% (9169/10000)

;;; Sparse

(defparameter mnist-train.sp (read-libsvm-data-sparse-multiclass "/home/wiz/tmp/mnist.scale"))
(defparameter mnist-test.sp  (read-libsvm-data-sparse-multiclass "/home/wiz/tmp/mnist.scale.t"))

;; Add 1 to labels because the labels of this dataset begin from 0
(dolist (datum mnist-train.sp) (incf (car datum)))
(dolist (datum mnist-test.sp)  (incf (car datum)))

(ql:quickload :clgplot)
(clgp:plot-histogram (mapcar (lambda (d) (clol.vector::sparse-vector-length (cdr d)))
                             mnist-train.sp) 20)

(defparameter mnist-arow.sp (make-one-vs-one mnist-dim 10 'sparse-arow 10d0))
(time (loop repeat 8 do (train mnist-arow.sp mnist-train.sp)))

;; Evaluation took:
;;   1.347 seconds of real time
;;   1.348425 seconds of total run time (1.325365 user, 0.023060 system)
;;   [ Run times consist of 0.012 seconds GC time, and 1.337 seconds non-GC time. ]
;;   100.07% CPU
;;   4,570,387,768 processor cycles
;;   337,618,400 bytes consed

;;;;; news20 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter news20-dim 62060)
(defparameter news20-train (read-libsvm-data-sparse-multiclass "/home/wiz/datasets/news20.scale"))
(defparameter news20-test (read-libsvm-data-sparse-multiclass "/home/wiz/datasets/news20.t.scale"))

(defparameter news20-arow (make-one-vs-one news20-dim 20 'sparse-arow 1d0))

(time (train news20-arow news20-train))
(time (test news20-arow news20-test))

(loop repeat 10 do
  (train news20-arow news20-train)
  (test news20-arow news20-test))

(defparameter news20-arow-1vr (make-one-vs-rest news20-dim 20 'sparse-arow 10d0))

(time (train news20-arow-1vr news20-train))
(time (test news20-arow-1vr news20-test))

(loop repeat 10 do
  (train news20-arow-1vr news20-train)
  (test news20-arow-1vr news20-test))

(time (loop repeat 10 do
  (train news20-arow-1vr news20-train)))

;;; Almost data dimension < 500
;; (ql:quickload :clgplot)
;; (clgp:plot-histogram
;;  (mapcar (lambda (x)
;;            (clol.vector::sparse-vector-length (cdr x)))
;;          news20-train)
;;  200)

;;;;; news20.binary ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defparameter news20.binary-dim 1355191)
(defparameter news20.binary (read-libsvm-data-sparse "/home/wiz/datasets/news20.binary"))
(defparameter news20.binary.arow (make-sparse-arow news20.binary-dim 10d0))
(time (loop repeat 20 do (train news20.binary.arow news20.binary)))
(test news20.binary.arow news20.binary)

(ql:quickload :clgplot)
(clgp:plot-histogram (mapcar (lambda (d) (clol.vector::sparse-vector-length (cdr d)))
                             news20.binary) 200 :x-range '(0 3000)
                             :output "/home/wiz/tmp/news20.binary-histogram.png")

;; Evaluation took:
;;   1.588 seconds of real time
;;   1.588995 seconds of total run time (1.582495 user, 0.006500 system)
;;   [ Run times consist of 0.006 seconds GC time, and 1.583 seconds non-GC time. ]
;;   100.06% CPU
;;   5,386,830,659 processor cycles
;;   59,931,648 bytes consed
  
;; Accuracy: 99.74495%, Correct: 19945, Total: 19996

;;; AROW++
;; wiz@prime:~/datasets$ arow_learn -i 20 news20.binary news20.binary.model.arow 
;; Number of features: 1355191
;; Number of examples: 19996
;; Number of updates:  37643
;; Done!
;; Time: 9.0135 sec.

;; wiz@prime:~/datasets$ arow_test news20.binary news20.binary.model.arow 
;; Accuracy 99.915% (19979/19996)
;; (Answer, Predict): (t,p):9986 (t,n):9993 (f,p):4 (f,n):13
;; Done!
;; Time: 2.2762 sec.

;;; liblinear
;; wiz@prime:~/datasets$ time liblinear-train -q news20.binary news20.binary.model

;; real    0m2.800s
;; user    0m2.772s
;; sys     0m0.265s
;; wiz@prime:~/datasets$ liblinear-predict news20.binary news20.binary.model news20.binary.out
;; Accuracy = 99.875% (19971/19996)

;;;;; cod-rna ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Dence data

;; of classes: 2
;; of data: 59,535 / 271617 (validation) / 157413 (unused/remaining)
;; of features: 8

(defparameter cod-rna-dim 8)
(defparameter cod-rna (read-libsvm-data "/home/wiz/datasets/cod-rna" cod-rna-dim))
(defparameter cod-rna.t (read-libsvm-data "/home/wiz/datasets/cod-rna.t" cod-rna-dim))

;; it seem require scaling

(defparameter cod-rna.scale (read-libsvm-data "/home/wiz/datasets/cod-rna.scale" cod-rna-dim))
(defparameter cod-rna.t.scale (read-libsvm-data "/home/wiz/datasets/cod-rna.t.scale" cod-rna-dim))

(defparameter cod-rna-arow (make-arow cod-rna-dim 10d0))
(time (loop repeat 20 do (train cod-rna-arow cod-rna)))
(test cod-rna-arow cod-rna.t)

(defparameter cod-rna-arow (make-arow cod-rna-dim 0.1d0))
(loop repeat 20 do
  (train cod-rna-arow cod-rna)
  (test cod-rna-arow cod-rna.t))

(defparameter cod-rna-sgd (make-sgd cod-rna-dim 0.000000001d0 0.001d0))
(loop repeat 20 do
  (train cod-rna-sgd cod-rna)
  (test cod-rna-sgd cod-rna.t))

(defparameter cod-rna-adam (make-adam cod-rna-dim 0.00000000000000000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(loop repeat 20 do
  (train cod-rna-adam cod-rna)
  (test cod-rna-adam cod-rna.t))

(defparameter cod-rna-scw (make-scw cod-rna-dim 0.9d0 0.1d0))
(loop repeat 20 do
  (train cod-rna-scw cod-rna)
  (test cod-rna-scw cod-rna.t))

;; Evaluation took:
;;   0.989 seconds of real time
;;   0.991686 seconds of total run time (0.987975 user, 0.003711 system)
;;   [ Run times consist of 0.038 seconds GC time, and 0.954 seconds non-GC time. ]
;;   100.30% CPU
;;   3,356,870,825 processor cycles
;;   1,508,573,152 bytes consed
  
;; Accuracy: 79.89706%, Correct: 217014, Total: 271617

;; wiz@prime:~/datasets$ arow_learn -i 100 cod-rna cod-rna.model.arow
;; Number of features: 8
;; Number of examples: 59535
;; Number of updates:  5903680
;; Done!
;; Time: 9.0629 sec.

;; wiz@prime:~/datasets$ arow_test cod-rna.t cod-rna.model.arow
;; Accuracy 75.503% (205079/271617)
;; (Answer, Predict): (t,p):24835 (t,n):180244 (f,p):834 (f,n):65704
;; Done!
;; Time: 0.6416 sec.

(defparameter cod-rna.sp (read-libsvm-data-sparse "/home/wiz/datasets/cod-rna"))
(defparameter cod-rna.t.sp (read-libsvm-data-sparse "/home/wiz/datasets/cod-rna.t"))

;; it seem require scaling => use libsvm scaling

(defparameter cod-rna-arow.sp (make-sparse-arow cod-rna-dim 10d0))
(time (loop repeat 100 do (train cod-rna-arow.sp cod-rna.sp)))

;; Evaluation took:
;;   1.170 seconds of real time
;;   1.172760 seconds of total run time (1.172760 user, 0.000000 system)
;;   [ Run times consist of 0.040 seconds GC time, and 1.133 seconds non-GC time. ]
;;   100.26% CPU
;;   3,968,874,560 processor cycles
;;   1,508,573,184 bytes consed
