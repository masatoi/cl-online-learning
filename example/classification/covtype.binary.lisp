;;; -*- coding:utf-8; mode:lisp -*-

;;; covtype.binary: Binary classification
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter covtype-dim 54)
(defparameter covtype (read-data "/home/wiz/datasets/covtype.libsvm.binary.scale" covtype-dim))
(defparameter covtype-data
  (mapcar (lambda (datum)
            (if (= (car datum) 1.0d0)
                (cons -1d0 (cdr datum))
                (cons 1d0 (cdr datum))))
          covtype))

(defparameter arow-learner (make-arow covtype-dim 10d0))
(loop repeat 100 do
  (train arow-learner covtype-data)
  (test arow-learner covtype-data))

(defparameter scw-learner (make-scw covtype-dim 0.99d0 0.01d0))
(loop repeat 100 do
  (train scw-learner covtype-data)
  (test scw-learner covtype-data))

(defparameter lr-adam-learner (make-lr+adam covtype-dim 0.000001d0 0.001d0 1.d-8 0.9d0 0.99d0))
(loop repeat 10 do
  (train lr-adam-learner covtype-data)
  (test lr-adam-learner covtype-data))
