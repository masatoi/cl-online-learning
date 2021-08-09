;;; -*- coding:utf-8; mode:lisp -*-

;;; Iris: Multiclass classification, Dence data
;; dataset: 
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#iris

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter iris-dim 4)
(defparameter iris-class 3)

(defparameter iris
  (shuffle-vector
   (coerce (read-data "/home/wiz/datasets/iris.scale" iris-dim :multiclass-p t)
	   'simple-vector)))

(defparameter iris-train (subseq iris 0 100))
(defparameter iris-test (subseq iris 100))

;; Perceptron, 1 vs 1
(defparameter mul-percep (make-one-vs-rest iris-dim iris-class 'perceptron))

(loop repeat 2 do
  (train mul-percep iris)
  (test mul-percep iris-test))
;; Accuracy: 94.0%, Correct: 141, Total: 150

;; AROW, 1 vs rest
(defparameter mul-arow-1vr (make-one-vs-rest iris-dim iris-class 'arow 0.1))
(train mul-arow-1vr iris-train)
(test mul-arow-1vr iris-test)
;; Accuracy: 96.0%, Correct: 48, Total: 50

;; AROW, 1 vs 1
(defparameter mul-arow-1v1 (make-one-vs-one iris-dim iris-class 'arow 0.1))
(train mul-arow-1v1 iris-train)
(test mul-arow-1v1 iris-test)

;; Accuracy: 96.0%, Correct: 48, Total: 50
