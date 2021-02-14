;; -*- coding:utf-8; mode:lisp -*-

;; $ ros install masatoi/clgplot
;; (ql:quickload :clgplot)

(in-package :cl-user)

(defpackage :cl-online-learning.examples
  (:use :cl :cl-online-learning :cl-online-learning.utils)
  (:nicknames :clol.exam))

(in-package :cl-online-learning.examples)

(ql:quickload :clgplot)

;;; Utilities

(defun random-uniform (start end)
  (+ (random (- end start)) start))

(defun random-normal (&key (mean 0.0) (sd 1.0))
  (let ((alpha (random 1.0))
	(beta  (random 1.0)))
    (+ (* sd
	  (sqrt (* -2 (log alpha)))
	  (sin (* 2 pi beta)))
       mean)))

(defun seq (start end &optional (by 1))
  (loop for x from start to end by by collect x))


;;; Make Dataset (Sine wave)

(defparameter x-lst
  (loop repeat 100
        collect (make-array 1 :element-type 'double-float
                              :initial-element (random-uniform (- pi) pi))))

(defparameter y-lst
  (mapcar (lambda (x)
            (+ (sin (aref x 0)) (random-normal :sd 0.1) 1))
          x-lst))

(defparameter train-dataset
  (mapcar #'cons y-lst x-lst))

(defparameter x-lst-test
  (let ((pi1 (coerce pi 'single-float)))
    (loop for x from (- pi1) to pi1 by 0.1 collect
        (make-array 1 :element-type 'single-float :initial-element x))))

(defparameter y-lst-test
  (mapcar (lambda (x)
            (+ (sin (aref x 0)) (random-normal :sd 0.1)))
          x-lst-test))

(defparameter test-dataset
  (mapcar #'cons y-lst-test x-lst-test))

;;; Define model, train, test

(defparameter rls1 (make-rls 1 0.99))
(train rls1 train-dataset)
(rls-test rls1 test-dataset)

(clgp:plots (list y-lst
                  (loop for x in x-lst-test
                        collect (rls-predict rls1 x)))
            :x-seqs (list (mapcar (lambda (x) (aref x 0)) x-lst)
                          (seq (- pi) pi 0.1))
            :style '(points lines))

