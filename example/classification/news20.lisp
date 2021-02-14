;;; -*- coding:utf-8; mode:lisp -*-

;;; news20: Multiclass classification, Many feature, Sparse data
;; dataset: 
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#news20

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter news20-dim 62060)
(defparameter news20-class 20)

(defparameter news20-train
  (read-data "/home/wiz/datasets/news20.scale" news20-dim :sparse-p t :multiclass-p t))
(defparameter news20-test
  (read-data "/home/wiz/datasets/news20.t.scale" news20-dim :sparse-p t :multiclass-p t))

(defparameter news20-arow (make-one-vs-one news20-dim news20-class 'sparse-arow 1d0))

(time (train news20-arow news20-train))
(time (test news20-arow news20-test))

(loop repeat 10 do
  (train news20-arow news20-train)
  (test news20-arow news20-test))

(defparameter news20-arow-1vr (make-one-vs-rest news20-dim news20-class 'sparse-arow 10d0))

(time (train news20-arow-1vr news20-train))

(loop repeat 10 do
  (train news20-arow-1vr news20-train)
  (test news20-arow-1vr news20-test))

(loop repeat 12 do
  (train news20-arow-1vr news20-train))

(test news20-arow-1vr news20-test)

;; Accuracy: 86.90208%, Correct: 3470, Total: 3993

(loop repeat 10 do
  (train news20-arow-1vr news20-train)
  (test news20-arow-1vr news20-test))

(defparameter news20-scw (make-one-vs-one news20-dim news20-class 'sparse-scw 0.9d0 1d0))

(loop repeat 10 do
  (train news20-scw news20-train)
  (test news20-scw news20-test))

;;; Almost data dimension < 500
;; (ql:quickload :clgplot)
;; (clgp:plot-histogram
;;  (mapcar (lambda (x)
;;            (clol.vector::sparse-vector-length (cdr x)))
;;          news20-train)
;;  200)
