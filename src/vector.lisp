;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.vector
  (:use :cl)
  (:nicknames :clol.vector)
  (:export
   :make-dvec :v+ :v- :v-scale :elementwise-product :inner-product
   :sparse-vector-length :sparse-vector-index-vector :sparse-vector-value-vector
   :make-sparse-vector :make-empty-sparse-vector
   :dv+sv :dv-sv :ds-inner-product
   :dence-pseudosparse-v+ :dence-pseudosparse-v- :dence-pseudosparse-v-scale
   :dence-sparse-elementwise-product :dence-pseudosparse-elementwise-product))

(in-package :cl-online-learning.vector)

;;; Dence vector operators

(defun make-dvec (input-dimension initial-element)
  (make-array input-dimension :element-type 'double-float :initial-element initial-element))

(defun v+ (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length x)) do
    (setf (aref result i) (+ (aref x i) (aref y i))))
  result)

(defun v- (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length x)) do
    (setf (aref result i) (- (aref x i) (aref y i))))
  result)

(defun v-scale (vec n result)
  (declare (type double-float n)
	   (type (simple-array double-float) vec result)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length vec)) do
    (setf (aref result i) (* n (aref vec i))))
  result)

(defun elementwise-product (x y result)
  (declare (type (simple-array double-float) x y result)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length x)) do
    (setf (aref result i) (* (aref x i) (aref y i))))
  result)

(declaim (ftype (function ((simple-array double-float) (simple-array double-float))
                          double-float)
                inner-product))
(defun inner-product (x y)
  (declare (type (simple-array double-float) x y)
           (optimize (speed 3) (safety 0)))
  (let ((result 0.0d0))
    (declare (type double-float result))
    (loop for i from 0 to (1- (length x)) do
      (incf result (* (aref x i) (aref y i))))
    result))

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

;; dence-x = dence-x + sparse-y
(defun dv+sv (dence-x sparse-y)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (sparse-vector-length sparse-y)) do
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref dence-x dence-index)
            (+ (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i))))))

;; dence-x = dence-x - sparse-y
(defun dv-sv (dence-x sparse-y)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (sparse-vector-length sparse-y)) do
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref dence-x dence-index)
            (- (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i))))))

(defun ds-inner-product (dence-x sparse-y)
  (declare (type sparse-vector sparse-y)
           (type (simple-array double-float) dence-x)
           (optimize (speed 3) (safety 0)))
  (let ((result 0.0d0))
    (declare (type double-float result))
    (loop for i from 0 to (1- (sparse-vector-length sparse-y)) do
      (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
        (declare (type fixnum dence-index))
        (incf result
              (* (aref dence-x dence-index)
                 (aref (sparse-vector-value-vector sparse-y) i)))))
    result))

(defun dence-pseudosparse-v+ (dence-x pseudosparse-y index-vector result)
  (declare (type (simple-array double-float) dence-x pseudosparse-y result)
           (type (simple-array fixnum) index-vector)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length index-vector)) do
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (+ (aref dence-x dence-index)
               (aref pseudosparse-y dence-index)))))
  result)

(defun dence-pseudosparse-v- (dence-x pseudosparse-y index-vector result)
  (declare (type (simple-array double-float) dence-x pseudosparse-y result)
           (type (simple-array fixnum) index-vector)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length index-vector)) do
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (- (aref dence-x dence-index)
               (aref pseudosparse-y dence-index)))))
  result)

(defun dence-pseudosparse-v-scale (pseudosparse-x alpha index-vector result)
  (declare (type (simple-array double-float) pseudosparse-x result)
           (type (simple-array fixnum) index-vector)
           (type double-float alpha)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length index-vector)) do
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* alpha
               (aref pseudosparse-x dence-index)))))
  result)

(defun dence-sparse-elementwise-product (dence-x sparse-y result)
  (declare (type (simple-array double-float) dence-x result)
           (type sparse-vector sparse-y)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (sparse-vector-length sparse-y)) do
    (let ((dence-index (aref (sparse-vector-index-vector sparse-y) i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref dence-x dence-index)
               (aref (sparse-vector-value-vector sparse-y) i)))))
  result)

(defun dence-pseudosparse-elementwise-product (dence-x pseudosparse-y index-vector result)
  (declare (type (simple-array double-float) dence-x pseudosparse-y result)
           (type (simple-array fixnum) index-vector)
           (optimize (speed 3) (safety 0)))
  (loop for i from 0 to (1- (length index-vector)) do
    (let ((dence-index (aref index-vector i)))
      (declare (type fixnum dence-index))
      (setf (aref result dence-index)
            (* (aref dence-x dence-index)
               (aref pseudosparse-y dence-index)))))
  result)
