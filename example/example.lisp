;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;; Examples ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defpackage :cl-online-learning.struct.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :cl-ol.st.exam))

(in-package :cl-online-learning.struct.examples)

(defparameter a1a-dim 123)
(defparameter a1a-train (read-libsvm-data "/home/wiz/tmp/a1a" a1a-dim))
(defparameter a1a-test (read-libsvm-data "/home/wiz/tmp/a1a.t" a1a-dim))

(defparameter perceptron-learner (make-perceptron a1a-dim))
(time (loop repeat 100 do (perceptron-train perceptron-learner a1a-train)))
(perceptron-test perceptron-learner a1a-test)

;; Evaluation took:
;;   0.033 seconds of real time
;;   0.032494 seconds of total run time (0.032494 user, 0.000000 system)
;;   96.97% CPU
;;   110,237,150 processor cycles
;;   8,257,536 bytes consed

;; Accuracy: 82.416985%, Correct: 25513, Total: 30956

(defparameter one-vs-rest-learner (make-one-vs-rest 10 5 'arow 10d0))
(defparameter perceptron-learner (make-perceptron a1a-dim))
(time (loop repeat 100 do (perceptron-train perceptron-learner a1a-train)))
(perceptron-test perceptron-learner a1a-test)

(defparameter a9a-dim 123)
(defparameter a9a-train (read-libsvm-data "/home/wiz/tmp/a9a" a9a-dim))
(defparameter a9a-test (read-libsvm-data "/home/wiz/tmp/a9a.t" a9a-dim))

(defparameter perceptron-learner (make-perceptron a9a-dim))
(time (loop repeat 1000 do (train perceptron-learner a9a-train)))
(test perceptron-learner a9a-test)

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

(defparameter arow-learner (make-arow a9a-dim 10d0))
(time (loop repeat 1000 do (train arow-learner a9a-train)))
(test arow-learner a9a-test)

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
;;   15.106 seconds of real time
;;   15.115968 seconds of total run time (15.108628 user, 0.007340 system)
;;   [ Run times consist of 0.069 seconds GC time, and 15.047 seconds non-GC time. ]
;;   100.07% CPU
;;   51,243,139,014 processor cycles
;;   5,881,064,032 bytes consed

;;; CLOS version
;; Evaluation took:
;;   31.187 seconds of real time
;;   31.198607 seconds of total run time (31.189536 user, 0.009071 system)
;;   [ Run times consist of 0.140 seconds GC time, and 31.059 seconds non-GC time. ]
;;   100.04% CPU
;;   105,792,514,974 processor cycles
;;   5,881,026,464 bytes consed

(defparameter scw1-learner (make-scw1 a9a-dim 0.9d0 0.1d0))
(time (loop repeat 1000 do (train scw1-learner a9a-train)))
(test scw1-learner a9a-test)

;; Evaluation took:
;;   13.235 seconds of real time
;;   13.247439 seconds of total run time (13.239742 user, 0.007697 system)
;;   [ Run times consist of 0.093 seconds GC time, and 13.155 seconds non-GC time. ]
;;   100.09% CPU
;;   44,896,182,054 processor cycles
;;   7,375,026,592 bytes consed

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


;; inline

;; Evaluation took:
;;   4.897 seconds of real time
;;   4.900393 seconds of total run time (4.882436 user, 0.017957 system)
;;   [ Run times consist of 0.065 seconds GC time, and 4.836 seconds non-GC time. ]
;;   100.06% CPU
;;   16,611,290,450 processor cycles
;;   1,674,403,728 bytes consed
  
;; Accuracy: 79.72483%, Correct: 12980, Total: 16281
;; Evaluation took:
;;   15.734 seconds of real time
;;   15.743037 seconds of total run time (15.715719 user, 0.027318 system)
;;   [ Run times consist of 0.070 seconds GC time, and 15.674 seconds non-GC time. ]
;;   100.06% CPU
;;   53,370,435,311 processor cycles
;;   5,881,035,456 bytes consed
  
;; Accuracy: 84.964066%, Correct: 13833, Total: 16281
;; Evaluation took:
;;   13.888 seconds of real time
;;   13.900793 seconds of total run time (13.865083 user, 0.035710 system)
;;   [ Run times consist of 0.097 seconds GC time, and 13.804 seconds non-GC time. ]
;;   100.09% CPU
;;   47,111,367,519 processor cycles
;;   7,457,269,824 bytes consed

;; no inline

;; Accuracy: 79.72483%, Correct: 12980, Total: 16281
;; Evaluation took:
;;   4.855 seconds of real time
;;   4.858373 seconds of total run time (4.854418 user, 0.003955 system)
;;   [ Run times consist of 0.018 seconds GC time, and 4.841 seconds non-GC time. ]
;;   100.06% CPU
;;   16,468,458,253 processor cycles
;;   1,674,411,648 bytes consed
  
;; Accuracy: 79.72483%, Correct: 12980, Total: 16281
;; Evaluation took:
;;   14.896 seconds of real time
;;   14.904901 seconds of total run time (14.892879 user, 0.012022 system)
;;   [ Run times consist of 0.072 seconds GC time, and 14.833 seconds non-GC time. ]
;;   100.06% CPU
;;   50,530,172,355 processor cycles
;;   5,881,030,720 bytes consed
  
;; Accuracy: 84.964066%, Correct: 13833, Total: 16281
;; Evaluation took:
;;   13.324 seconds of real time
;;   13.333265 seconds of total run time (13.317252 user, 0.016013 system)
;;   [ Run times consist of 0.089 seconds GC time, and 13.245 seconds non-GC time. ]
;;   100.07% CPU
;;   45,196,042,976 processor cycles
;;   7,375,022,144 bytes consed

;; Accuracy: 79.72483%, Correct: 12980, Total: 16281
;; Evaluation took:
;;   4.756 seconds of real time
;;   4.759471 seconds of total run time (4.759471 user, 0.000000 system)
;;   [ Run times consist of 0.017 seconds GC time, and 4.743 seconds non-GC time. ]
;;   100.06% CPU
;;   16,134,008,169 processor cycles
;;   1,674,411,648 bytes consed
  
;; Accuracy: 79.72483%, Correct: 12980, Total: 16281
;; Evaluation took:
;;   14.781 seconds of real time
;;   14.791067 seconds of total run time (14.791067 user, 0.000000 system)
;;   [ Run times consist of 0.067 seconds GC time, and 14.725 seconds non-GC time. ]
;;   100.07% CPU
;;   50,138,829,951 processor cycles
;;   5,881,030,720 bytes consed
  
;; Accuracy: 84.964066%, Correct: 13833, Total: 16281
;; Evaluation took:
;;   13.408 seconds of real time
;;   13.417955 seconds of total run time (13.386376 user, 0.031579 system)
;;   [ Run times consist of 0.095 seconds GC time, and 13.323 seconds non-GC time. ]
;;   100.07% CPU
;;   45,480,948,294 processor cycles
;;   7,375,022,016 bytes consed


;;; Multi class

(defparameter iris-dim 4)

(defparameter iris
  (shuffle-vector
   (coerce (read-libsvm-data-multiclass "/home/wiz/tmp/iris.scale" iris-dim)
	   'simple-vector)))

(defparameter iris-train (subseq iris 0 100))
(defparameter iris-test (subseq iris 100))

;; one-vs-rest
(defparameter mulc (make-one-vs-rest 780 10 'arow 10d0))
(time (loop repeat 100 do (train mulc cl-ol.exam::mnist+1)))
(test mulc cl-ol.exam::mnist.t+1)

(defparameter mulc (make-one-vs-one 780 10 'arow 10d0))
(time (loop repeat 100 do (train mulc cl-ol.exam::mnist+1)))
(test mulc cl-ol.exam::mnist.t+1)

;; AROW
(defparameter mul-arow (make-one-vs-rest 4 3 'arow 0.1d0))
(train mul-arow iris-train)
(test mul-arow iris-test)

(defparameter mul-arow (make-one-vs-one 4 3 'arow 0.1d0))
(train mul-arow iris-train)
(test mul-arow iris-test)

;;; news20
(defparameter news20 (read-libsvm-data-multiclass "/home/wiz/datasets/news20.scale" 62060))

(defparameter news20
  (coerce (read-libsvm-data-multiclass "/home/wiz/datasets/news20.scale" 4)
          'simple-vector)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(in-package :cl-user)

(defparameter a1a-dim 123)
(defparameter a1a-train (clol.utils:read-libsvm-data "/home/wiz/tmp/a1a" a1a-dim))
(defparameter a1a-test (clol.utils:read-libsvm-data "/home/wiz/tmp/a1a.t" a1a-dim))

(defparameter perceptron-learner (clol:make-perceptron a1a-dim))
(time (loop repeat 100 do (clol:perceptron-train perceptron-learner a1a-train)))
(clol:test perceptron-learner a1a-test)

(clol:perceptron-update perceptron-learner (cdadr a1a-test) (caadr a1a-test))
(clol:perceptron-predict perceptron-learner (cdadr a1a-test))

(defparameter iris
  (clol.utils:shuffle-vector
   (coerce (clol.utils:read-libsvm-data-multiclass "/home/wiz/tmp/iris.scale" 4)
	   'simple-vector)))


(defparameter iris-dim 4)

(defparameter iris
  (clol.utils:shuffle-vector
   (coerce (clol.utils:read-libsvm-data-multiclass
            (merge-pathnames #P"t/dataset/iris.scale"
                             (asdf:system-source-directory :cl-online-learning))
            iris-dim)
	   'simple-vector)))

(defparameter iris-train (loop for i from 0 to (1- 100) collect (svref iris i)))
(defparameter iris-test (loop for i from 100 to (1- 150) collect (svref iris i)))

(defparameter arow-1-vs-rest
  (clol:make-one-vs-rest iris-dim     ; Input data dimension
                         3            ; Number of class
                         'arow 10d0)) ; Binary classifier type and its parameters
(clol:train mul-arow iris-train)
(clol:test mul-arow iris-test)

(defparameter mul-arow (clol:make-one-vs-one 4 3 'arow 10d0))
(clol:train mul-arow cl-ol.exam::iris-train)
(clol:test mul-arow cl-ol.exam::iris-test)

(defparameter mulc (clol:make-one-vs-one 780 10 'arow 10d0))
(time (loop repeat 10 do (clol:train mulc cl-ol.exam::mnist+1)))
(clol:test mulc cl-ol.exam::mnist.t+1)
