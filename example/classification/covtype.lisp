;;; -*- coding:utf-8; mode:lisp -*-

;;; covtype: Multiclass classification
;; dataset:
;; https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#covtype

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(defparameter covtype-dim 54)
(defparameter covtype-n-class 7)
(defparameter covtype-train (clol.utils:read-data "/home/wiz/datasets/covtype.scale" covtype-dim
                                                  :multiclass-p t))
(defparameter covtype-test (clol.utils:read-data "/home/wiz/datasets/covtype.scale.t" covtype-dim
                                                :multiclass-p t))


(defparameter arow-learner (make-one-vs-one covtype-dim covtype-n-class 'arow 10.0))
(loop repeat 5 do
  (train arow-learner covtype-train)
  (test arow-learner covtype-test))

(defparameter scw-learner (make-one-vs-one covtype-dim covtype-n-class 'scw 0.99 0.01))
(loop repeat 5 do
  (train scw-learner covtype-train)
  (test scw-learner covtype-test))

;; (clrf::write-to-r-format-from-clol-dataset covtype-train "/home/wiz/datasets/covtype-for-r")
