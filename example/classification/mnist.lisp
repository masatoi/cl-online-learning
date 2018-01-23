;;; -*- coding:utf-8; mode:lisp -*-

;;; MNIST: Multiclass classification, Dence/Sparse data, More bigger data
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter mnist-dim 784)
(defparameter mnist-class 10)
(defparameter mnist-train (read-data "/home/wiz/datasets/mnist.scale" mnist-dim :multiclass-p t))
(defparameter mnist-test  (read-data "/home/wiz/datasets/mnist.scale.t" mnist-dim :multiclass-p t))

;; Add 1 to labels because the labels of this dataset begin from 0
(dolist (datum mnist-train) (incf (car datum)))
(dolist (datum mnist-test)  (incf (car datum)))

;; 11sec Accuracy: 94.65%, Correct: 9465, Total: 10000
(defparameter mnist-arow (make-one-vs-one mnist-dim mnist-class 'arow 10d0))
(time (loop repeat 10 do
  (train mnist-arow mnist-train)
  (test mnist-arow mnist-test)))

(defparameter mnist-arow (make-one-vs-rest mnist-dim mnist-class 'arow 10d0))
(time (loop repeat 10 do
  (train mnist-arow mnist-train)
  (test mnist-arow mnist-test)))

(defparameter mnist-scw (make-one-vs-one mnist-dim mnist-class 'scw 0.9d0 0.1d0))
(time (loop repeat 10 do
  (train mnist-scw mnist-train)
  (test mnist-scw mnist-test)))

;; 10 sec Accuracy: 91.78%, Correct: 9178, Total: 10000
(defparameter mnist-perceptron (make-one-vs-one mnist-dim mnist-class 'perceptron))
(time (loop repeat 10 do
  (train mnist-perceptron mnist-train)
  (test mnist-perceptron mnist-test)))

(defparameter mnist-sgd (make-one-vs-one mnist-dim mnist-class 'lr+sgd 0.00001d0 0.01d0))
(time (loop repeat 10 do
  (train mnist-sgd mnist-train)
  (test mnist-sgd mnist-test)))

(defparameter mnist-adam (make-one-vs-one mnist-dim mnist-class 'lr+adam 0.001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(time (loop repeat 10 do
  (train mnist-adam mnist-train)
  (test mnist-adam mnist-test)))

(require :sb-sprof)

(sb-sprof:with-profiling (:max-samples 10000
				       :mode :alloc
				       :report :flat)
  (time (loop repeat 10 do
    (train mnist-lr-sgd mnist-train)
    (test mnist-lr-sgd mnist-test))))

;; 20 sec Accuracy: 93.87%, Correct: 9387, Total: 10000
(defparameter mnist-lr-sgd (make-one-vs-one mnist-dim mnist-class 'sgd 0.00001d0 0.01d0))
(time (loop repeat 10 do
  (train mnist-lr-sgd mnist-train)
  (test mnist-lr-sgd mnist-test)))

;;            Self        Total        Cumul
;;   Nr  Count     %  Count     %  Count     %    Calls  Function
;; ------------------------------------------------------------------------
;;    1   3693  36.9   3693  36.9   3693  36.9        -  SB-KERNEL:TWO-ARG-*
;;    2   1834  18.3   1834  18.3   5527  55.3        -  SB-KERNEL:TWO-ARG-+
;;    3   1306  13.1   1306  13.1   6833  68.3        -  CL-ONLINE-LEARNING.VECTOR:DOT
;;    4   1054  10.5   1054  10.5   7887  78.9        -  SB-KERNEL:%NEGATE
;;    5   1051  10.5   7386  73.9   8938  89.4        -  CL-ONLINE-LEARNING::LOGISTIC-REGRESSION-GRADIENT
;;    6    530   5.3    530   5.3   9468  94.7        -  CL-ONLINE-LEARNING::SIGMOID
;;    7    529   5.3    529   5.3   9997 100.0        -  SB-KERNEL:TWO-ARG--
;;    8      3   0.0      3   0.0  10000 100.0        -  "foreign function pthread_sigmask"
;;    9      0   0.0  10000 100.0  10000 100.0        -  (LAMBDA NIL :IN #:DROP-THRU-TAG-1)
;;   10      0   0.0  10000 100.0  10000 100.0        -  SB-EXT:CALL-WITH-TIMING
;;   11      0   0.0  10000 100.0  10000 100.0        -  "Unknown component: #x1022114C50"
;;   12      0   0.0  10000 100.0  10000 100.0        -  SB-INT:SIMPLE-EVAL-IN-LEXENV
;;   13      0   0.0  10000 100.0  10000 100.0        -  EVAL
;;   14      0   0.0  10000 100.0  10000 100.0        -  (LAMBDA NIL :IN SWANK:INTERACTIVE-EVAL)
;;   15      0   0.0  10000 100.0  10000 100.0        -  SWANK::CALL-WITH-RETRY-RESTART
;;   16      0   0.0  10000 100.0  10000 100.0        -  SWANK::CALL-WITH-BUFFER-SYNTAX
;;   17      0   0.0  10000 100.0  10000 100.0        -  SWANK:EVAL-FOR-EMACS
;;   18      0   0.0  10000 100.0  10000 100.0        -  (LAMBDA NIL :IN SWANK::SPAWN-WORKER-THREAD)
;;   19      0   0.0  10000 100.0  10000 100.0        -  SWANK/SBCL::CALL-WITH-BREAK-HOOK
;;   20      0   0.0  10000 100.0  10000 100.0        -  (FLET SWANK/BACKEND:CALL-WITH-DEBUGGER-HOOK :IN "/home/wiz/.roswell/lisp/quicklisp/dists/quicklisp/software/slime-v2.18/swank/sbcl.lisp")
;;   21      0   0.0  10000 100.0  10000 100.0        -  SWANK::CALL-WITH-BINDINGS
;;   22      0   0.0  10000 100.0  10000 100.0        -  (FLET #:WITHOUT-INTERRUPTS-BODY-1158 :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   23      0   0.0  10000 100.0  10000 100.0        -  (FLET SB-THREAD::WITH-MUTEX-THUNK :IN SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE)
;;   24      0   0.0  10000 100.0  10000 100.0        -  (FLET #:WITHOUT-INTERRUPTS-BODY-359 :IN SB-THREAD::CALL-WITH-MUTEX)
;;   25      0   0.0  10000 100.0  10000 100.0        -  SB-THREAD::CALL-WITH-MUTEX
;;   26      0   0.0  10000 100.0  10000 100.0        -  SB-THREAD::INITIAL-THREAD-FUNCTION-TRAMPOLINE
;;   27      0   0.0  10000 100.0  10000 100.0        -  "foreign function call_into_lisp"
;;   28      0   0.0  10000 100.0  10000 100.0        -  "foreign function new_thread_trampoline"
;;   29      0   0.0   8444  84.4  10000 100.0        -  SGD-UPDATE
;;   30      0   0.0   8444  84.4  10000 100.0        -  ONE-VS-ONE-UPDATE
;;   31      0   0.0   8444  84.4  10000 100.0        -  ONE-VS-ONE-TRAIN
;;   32      0   0.0   1556  15.6  10000 100.0        -  SGD-PREDICT
;;   33      0   0.0   1556  15.6  10000 100.0        -  ONE-VS-ONE-PREDICT
;;   34      0   0.0   1556  15.6  10000 100.0        -  (LAMBDA (CL-ONLINE-LEARNING::DATUM) :IN ONE-VS-ONE-TEST)
;;   35      0   0.0   1556  15.6  10000 100.0        -  COUNT-IF
;;   36      0   0.0   1556  15.6  10000 100.0        -  ONE-VS-ONE-TEST
;;   37      0   0.0      3   0.0  10000 100.0        -  "foreign function interrupt_handle_pending"
;;   38      0   0.0      3   0.0  10000 100.0        -  "foreign function handle_trap"
;; ------------------------------------------------------------------------
;;           0   0.0                                     elsewhere

;; 100 sec Accuracy: 94.29%, Correct: 9429, Total: 10000
(defparameter mnist-lr-adam (make-one-vs-one mnist-dim mnist-class 'adam 0.000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(time (loop repeat 10 do
  (train mnist-lr-adam mnist-train)
  (test mnist-lr-adam mnist-test)))

(require :sb-sprof)

(sb-sprof:with-profiling (:max-samples 10000
				       :mode :alloc
				       :report :flat)
  (time (loop repeat 10 do
    (train mnist-lr-adam mnist-train)
    (test mnist-lr-adam mnist-test)))
  )

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

(defparameter mnist-train.sp
  (read-data "/home/wiz/tmp/mnist.scale" mnist-dim :sparse-p t :multiclass-p t))
(defparameter mnist-test.sp
  (read-data "/home/wiz/tmp/mnist.scale.t" mnist-dim :sparse-p t :multiclass-p t))

;; Add 1 to labels because the labels of this dataset begin from 0
(dolist (datum mnist-train.sp) (incf (car datum)))
(dolist (datum mnist-test.sp)  (incf (car datum)))

(defparameter mnist-arow.sp (make-one-vs-one mnist-dim mnist-class 'sparse-arow 10d0))
(time (loop repeat 8 do (train mnist-arow.sp mnist-train.sp)))

;; Evaluation took:
;;   1.347 seconds of real time
;;   1.348425 seconds of total run time (1.325365 user, 0.023060 system)
;;   [ Run times consist of 0.012 seconds GC time, and 1.337 seconds non-GC time. ]
;;   100.07% CPU
;;   4,570,387,768 processor cycles
;;   337,618,400 bytes consed

;; Evaluation took:
;;   1.156 seconds of real time
;;   1.156480 seconds of total run time (1.156480 user, 0.000000 system)
;;   100.00% CPU
;;   3,922,998,785 processor cycles
;;   20,383,344 bytes consed
