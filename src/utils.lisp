;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.utils
  (:use :cl :cl-online-learning.vector)
  (:nicknames :clol.utils)
  (:export :shuffle-vector :read-data))

(in-package :cl-online-learning.utils)

;; Fisher–Yates shuffle
(defun shuffle-vector (vec)
  (loop for i from (1- (length vec)) downto 1 do
    (let* ((j (random (1+ i)))
	   (tmp (svref vec i)))
      (setf (svref vec i) (svref vec j))
      (setf (svref vec j) tmp)))
  vec)

;;; Read dataset

;; Read libsvm dataset
(defun make-datum-pair (read-data data-dimension &key multiclass-p)
  (let* ((input-vector (make-array data-dimension :element-type 'double-float :initial-element 0d0))
         (read-data-split (split-sequence:split-sequence #\Space read-data :test #'char=))
         (training-label (if multiclass-p
                           (1- (parse-integer (car read-data-split)))
                           (coerce (parse-number:parse-number (car read-data-split)) 'double-float))))
    (dolist (index-num-str (cdr read-data-split))
      (multiple-value-bind (lst len)
          (split-sequence:split-sequence #\: index-num-str :test #'char=)
        (when (not (zerop len))
          (setf (aref input-vector (1- (parse-integer (car lst))))
                (coerce (parse-number:parse-number (cadr lst)) 'double-float)))))
    (cons training-label input-vector)))

(defun make-index-num-alist (read-data-split)
  (labels ((iter (pair-str-list product)
             (if (null pair-str-list)
               (nreverse product)
               (multiple-value-bind (lst len)
                   (split-sequence:split-sequence #\: (car pair-str-list) :test #'char=)
                 (if (not (zerop len))
                   (iter (cdr pair-str-list)
                         (cons
                          (list (1- (parse-integer (car lst)))
                                (coerce (parse-number:parse-number (cadr lst)) 'double-float))
                          product))
                   (iter (cdr pair-str-list) product))))))
    (iter (cdr read-data-split) nil)))
                 
(defun make-datum-pair-sparse (read-data data-dimension &key multiclass-p)
  (declare (ignore data-dimension))
  (let* ((read-data-split (split-sequence:split-sequence #\Space read-data :test #'char=))
         (training-label (if multiclass-p
                           (1- (parse-integer (car read-data-split)))
                           (coerce (parse-number:parse-number (car read-data-split)) 'double-float)))
         (index-num-alist (make-index-num-alist read-data-split))
         (sparse-dim (length index-num-alist))
         (input-vector (make-empty-sparse-vector sparse-dim)))
    (loop for i from 0 to (1- sparse-dim)
          for index-num-pair in index-num-alist
          do
       (setf (aref (sparse-vector-index-vector input-vector) i) (car index-num-pair)
             (aref (sparse-vector-value-vector input-vector) i) (cadr index-num-pair)))
    (cons training-label input-vector)))

(defun read-data (data-path data-dimension &key sparse-p multiclass-p)
  (with-open-file (f data-path :direction :input)
    (labels
        ((read-loop (data-list)
           (let ((read-data (read-line f nil nil)))
             (if (null read-data)
               (nreverse data-list)
               (let ((datum-pair
                      (if sparse-p
                        (make-datum-pair-sparse read-data data-dimension :multiclass-p multiclass-p)
                        (make-datum-pair read-data data-dimension :multiclass-p multiclass-p))))
                 (read-loop (cons datum-pair data-list)))))))
      (read-loop nil))))
