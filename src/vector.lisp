;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.vector
  (:use :cl)
  (:nicknames :clol.vector)
  (:export
   :make-dvec :v+ :v- :v*n :v+n :v* :v/ :v-sqrt
   :dot :dot!
   :sparse-vector-length :sparse-vector-index-vector :sparse-vector-value-vector
   :make-sparse-vector :make-empty-sparse-vector
   :s-v*n :sps-v*n
   :ds-v+ :ds-v- :ds-v* :ds-v/ :ds-dot :ds-dot!
   :dps-v+ :dps-v- :ps-v*n :dps-v*))

(in-package :cl-online-learning.vector)

(defun make-dvec (input-dimension initial-element)
  (make-array input-dimension :element-type 'double-float :initial-element initial-element))

(defmacro dovec (vec var &body body)
  `(loop for ,var fixnum from 0 to (1- (length ,vec)) do ,@body))

;;; Dence vector operators

(defun v+ (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (+ (aref x i) (aref y i))))
  result)

(defun v- (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (- (aref x i) (aref y i))))
  result)

(defun v*n (vec n result)
  (declare (type double-float n)
	   (type (simple-array double-float) vec result)
           (optimize (speed 3) (safety 0)))
  (dovec vec i (setf (aref result i) (* n (aref vec i))))
  result)

(defun v+n (x n result)
  (declare (type double-float n)
           (type (simple-array double-float) x result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (+ (aref x i) n)))
  result)

(defun v* (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (* (aref x i) (aref y i))))
  result)

(defun v/ (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (/ (aref x i) (aref y i))))
  result)

(defun v-sqrt (x result)
  (declare (type (simple-array (double-float 0d0)) x result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (sqrt (aref x i))))
  result)

(declaim (ftype (function ((simple-array double-float) (simple-array double-float))
                          double-float)
                dot))
(defun dot (x y)
  (declare (type (simple-array double-float) x y)
           (optimize (speed 3) (safety 0)))
  (let ((result 0.0d0))
    (declare (type double-float result))
    (dovec x i (incf result (* (aref x i) (aref y i))))
    result))

(defun dot! (x y result)
  (declare (type (simple-array double-float) x y)
           (type (simple-array double-float 1) result)
           (optimize (speed 3) (safety 0)))
  (let ((acc 0d0))
    (declare (type double-float acc))
    (dovec x i (incf acc (* (aref x i) (aref y i))))
    (setf (aref result 0) acc))
  result)

;;; Sparse vector operators

(defstruct (sparse-vector (:constructor %make-sparse-vector))
  (length 0 :type fixnum)
  (index-vector #() :type (simple-array fixnum))
  (value-vector #() :type (simple-array double-float)))

(defun make-sparse-vector (index-vector value-vector)
  (assert (= (length index-vector) (length value-vector)))
  (%make-sparse-vector :length (length index-vector)
                       :index-vector index-vector
                       :value-vector value-vector))

(defun make-empty-sparse-vector (sparse-dim)
  (%make-sparse-vector :length sparse-dim
                       :index-vector (make-array sparse-dim :element-type 'fixnum)
                       :value-vector (make-array sparse-dim :element-type 'double-float)))

(defmacro dosvec (svec var &body body)
  `(loop for ,var fixnum from 0 to (1- (sparse-vector-length ,svec)) do ,@body))

(defun s-v*n (sparse-x n result)
  (declare (type sparse-vector sparse-x result)
           (type double-float n)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-x i
    (setf (aref (sparse-vector-value-vector result) i)
          (* (aref (sparse-vector-value-vector sparse-x) i) n)))
  result)

;; in case of result is pseudosparse-vector
(defun sps-v*n (sparse-x n result)
  (declare (type sparse-vector sparse-x)
           (type (simple-array double-float) result)
           (type double-float n)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-x i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-x) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref (sparse-vector-value-vector sparse-x) i) n))))
  result)

(defun ds-v+ (dence-x sparse-y result)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x result)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (+ (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun ds-v- (dence-x sparse-y result)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x result)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (- (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun ds-v* (dence-x sparse-y result)
  (declare (type (simple-array double-float) dence-x result)
           (type sparse-vector sparse-y)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun ds-v/ (dence-x sparse-y result)
  (declare (type (simple-array double-float) dence-x result)
           (type sparse-vector sparse-y)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (/ (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun ds-dot (dence-x sparse-y)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x)
           (optimize (speed 3) (safety 0)))
  (let ((result 0.0d0))
    (declare (type double-float result))
    (dosvec sparse-y i
      (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
        (declare (type fixnum dence-index))
        (incf result
              (* (aref dence-x dence-index)
                 (aref (sparse-vector-value-vector sparse-y) i)))))
    result))

(defun ds-dot! (dence-x sparse-y result)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x)
           (type (simple-array double-float 1) result)
           (optimize (speed 3) (safety 0)))
  (let ((acc 0d0))
    (declare (type double-float acc))
    (dosvec sparse-y i
      (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
        (declare (type fixnum dence-index))
        (incf acc
              (* (aref dence-x dence-index)
                 (aref (sparse-vector-value-vector sparse-y) i)))))
    (setf (aref result 0) acc)
    result))

;;; Use dence vector with index-vector of sparse-vector as sparse-vector (pseudosparse-vector)
(defun dps-v+ (dence-x pseudosparse-y index-vector result)
  (declare (type (simple-array double-float) dence-x pseudosparse-y result)
           (type (simple-array fixnum) index-vector)
           (optimize (speed 3) (safety 0)))
  (dovec index-vector i
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (+ (aref dence-x dence-index)
               (aref pseudosparse-y dence-index)))))
  result)

(defun dps-v- (dence-x pseudosparse-y index-vector result)
  (declare (type (simple-array double-float) dence-x pseudosparse-y result)
           (type (simple-array fixnum) index-vector)
           (optimize (speed 3) (safety 0)))
  (dovec index-vector i
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (- (aref dence-x dence-index)
               (aref pseudosparse-y dence-index)))))
  result)

(defun dps-v* (dence-x pseudosparse-y index-vector result)
  (declare (type (simple-array double-float) dence-x pseudosparse-y result)
           (type (simple-array fixnum) index-vector)
           (optimize (speed 3) (safety 0)))
  (dovec index-vector i
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref dence-x dence-index)
               (aref pseudosparse-y dence-index)))))
  result)

(defun ps-v*n (pseudosparse-x n index-vector result)
  (declare (type (simple-array double-float) pseudosparse-x result)
           (type (simple-array fixnum) index-vector)
           (type double-float n)
           (optimize (speed 3) (safety 0)))
  (dovec index-vector i
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* n (aref pseudosparse-x dence-index)))))
  result)
