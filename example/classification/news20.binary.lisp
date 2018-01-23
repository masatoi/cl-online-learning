;;; -*- coding:utf-8; mode:lisp -*-

;;; news20.binary: Binary classification, Sparse data
;; dataset: 
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#news20.binary

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter news20.binary-dim 1355191)
(defparameter news20.binary
  (read-data "/home/wiz/datasets/news20.binary" news20.binary-dim :sparse-p t))
(defparameter news20.binary.arow (make-sparse-arow news20.binary-dim 10d0))
(time (loop repeat 20 do (train news20.binary.arow news20.binary)))
(test news20.binary.arow news20.binary)

(defparameter news20.binary.lr+sgd (make-sparse-lr+sgd news20.binary-dim 0.00001d0 0.01d0))
(time (loop repeat 20 do (train news20.binary.lr+sgd news20.binary)))

(defparameter news20.binary-train (subseq news20.binary 0 15000))
(defparameter news20.binary-test (subseq news20.binary 15000 19996))

(defparameter news20.binary.lr+sgd (make-sparse-sgd news20.binary-dim 0.00001d0 0.01d0))

(progn
  (sparse-sgd-update news20.binary.lr+sgd (cdar news20.binary-train) (caar news20.binary-train))
  nil)

(train news20.binary.lr+sgd news20.binary-train)
(time (loop repeat 20 do (train news20.binary.lr+sgd news20.binary-train)))
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
