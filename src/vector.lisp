;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.vector
  (:use :cl)
  (:nicknames :clol.vector)
  (:export
   :make-vec :dovec :v+ :v- :v*n :v+n :v* :v/ :v-sqrt
   :dot :dot!
   :sparse-vector :sparse-vector-length :sparse-vector-index-vector :sparse-vector-value-vector
   :make-sparse-vector :make-empty-sparse-vector :dosvec
   :s-v*n :sps-v*n
   :ds-v+ :ds-v- :ds-v* :ds2s-v* :ds-v/ :ds-dot :ds-dot!
   :dps-v+ :dps-v- :ps-v*n :dps-v*))

(in-package :cl-online-learning.vector)

(defun make-vec (input-dimension initial-element)
  (make-array input-dimension :element-type 'single-float :initial-element initial-element))

(defmacro dovec (vec var &body body)
  `(loop for ,var fixnum from 0 below (length ,vec) do ,@body))

;;; Dence vector operators

(defun v+ (x y result)
  (declare (type (simple-array single-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (+ (aref x i) (aref y i))))
  result)

(defun v- (x y result)
  (declare (type (simple-array single-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (- (aref x i) (aref y i))))
  result)

(defun v*n (vec n result)
  (declare (type single-float n)
	   (type (simple-array single-float) vec result)
           (optimize (speed 3) (safety 0)))
  (dovec vec i (setf (aref result i) (* n (aref vec i))))
  result)

(defun v+n (x n result)
  (declare (type single-float n)
           (type (simple-array single-float) x result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (+ (aref x i) n)))
  result)

(defun v* (x y result)
  (declare (type (simple-array single-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (* (aref x i) (aref y i))))
  result)

(defun v/ (x y result)
  (declare (type (simple-array single-float) x y result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (/ (aref x i) (aref y i))))
  result)

(defun v-sqrt (x result)
  (declare (type (simple-array (single-float 0.0)) x result)
           (optimize (speed 3) (safety 0)))
  (dovec x i (setf (aref result i) (sqrt (aref x i))))
  result)

(declaim (ftype (function ((simple-array single-float) (simple-array single-float))
                          single-float)
                dot))
(defun dot (x y)
  (declare (type (simple-array single-float) x y)
           (optimize (speed 3) (safety 0)))
  (let ((result 0.0))
    (declare (type single-float result))
    (dovec x i (incf result (* (aref x i) (aref y i))))
    result))

(defun dot! (x y result)
  (declare (type (simple-array single-float) x y)
           (type (simple-array single-float 1) result)
           (optimize (speed 3) (safety 0)))
  (let ((acc 0.0))
    (declare (type single-float acc))
    (dovec x i (incf acc (* (aref x i) (aref y i))))
    (setf (aref result 0) acc))
  result)

;;; Sparse vector operators

(defstruct (sparse-vector (:constructor %make-sparse-vector))
  (length 0 :type fixnum)
  (index-vector #() :type (simple-array fixnum))
  (value-vector #() :type (simple-array single-float)))

(defun make-sparse-vector (index-vector value-vector)
  (assert (= (length index-vector) (length value-vector)))
  (%make-sparse-vector :length (length index-vector)
                       :index-vector index-vector
                       :value-vector value-vector))

(defun make-empty-sparse-vector (sparse-dim)
  (%make-sparse-vector :length sparse-dim
                       :index-vector (make-array sparse-dim :element-type 'fixnum)
                       :value-vector (make-array sparse-dim :element-type 'single-float)))

(defmacro dosvec (svec var &body body)
  `(loop for ,var fixnum from 0 below (sparse-vector-length ,svec) do ,@body))

(defun s-v*n (sparse-x n result)
  (declare (type sparse-vector sparse-x result)
           (type single-float n)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-x i
    (setf (aref (sparse-vector-value-vector result) i)
          (* (aref (sparse-vector-value-vector sparse-x) i) n)))
  result)

;; in case of result is pseudosparse-vector
(defun sps-v*n (sparse-x n result)
  (declare (type sparse-vector sparse-x)
           (type (simple-array single-float) result)
           (type single-float n)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-x i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-x) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref (sparse-vector-value-vector sparse-x) i) n))))
  result)

(defun ds-v+ (dence-x sparse-y result)
  (declare (type sparse-vector sparse-y)
           (type (simple-array single-float) dence-x result)
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
           (type (simple-array single-float) dence-x result)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (- (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun ds-v* (dence-x sparse-y result)
  (declare (type (simple-array single-float) dence-x result)
           (type sparse-vector sparse-y)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun ds2s-v* (dence-x sparse-y sparse-result)
  (declare (type (simple-array single-float) dence-x)
           (type sparse-vector sparse-y sparse-result)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref (sparse-vector-value-vector sparse-result) i)
            (* (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  sparse-result)

(defun ds-v/ (dence-x sparse-y result)
  (declare (type (simple-array single-float) dence-x result)
           (type sparse-vector sparse-y)
           (optimize (speed 3) (safety 0)))
  (dosvec sparse-y i
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (/ (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(declaim (ftype (function ((simple-array single-float) sparse-vector)
                          single-float)
                ds-dot))
(defun ds-dot (dence-x sparse-y)
  (declare (type sparse-vector sparse-y)
           (type (simple-array single-float) dence-x)
           (optimize (speed 3) (safety 0)))
  (let ((result 0.0))
    (declare (type single-float result))
    (dosvec sparse-y i
      (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
        (declare (type fixnum dence-index))
        (incf result
              (* (aref dence-x dence-index)
                 (aref (sparse-vector-value-vector sparse-y) i)))))
    result))

(defun ds-dot! (dence-x sparse-y result)
  (declare (type sparse-vector sparse-y)
           (type (simple-array single-float) dence-x)
           (type (simple-array single-float 1) result)
           (optimize (speed 3) (safety 0)))
  (let ((acc 0.0))
    (declare (type single-float acc))
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
  (declare (type (simple-array single-float) dence-x pseudosparse-y result)
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
  (declare (type (simple-array single-float) dence-x pseudosparse-y result)
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
  (declare (type (simple-array single-float) dence-x pseudosparse-y result)
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
  (declare (type (simple-array single-float) pseudosparse-x result)
           (type (simple-array fixnum) index-vector)
           (type single-float n)
           (optimize (speed 3) (safety 0)))
  (dovec index-vector i
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* n (aref pseudosparse-x dence-index)))))
  result)
