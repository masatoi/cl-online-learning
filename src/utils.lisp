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

;;; Autoscale

(defmacro doseq ((var seq) &body body)
  `(cond
     ((listp ,seq) (dolist (,var ,seq) ,@body))
     ((arrayp ,seq) (loop for ,var across ,seq do ,@body))
     (t (error "Error: seq must be list or array"))))

;; dataset is a sequence of pairs of target and input
(defun mean-vector (dataset)
  (let* ((datum-dim (length (cdr (elt dataset 0))))
         (data-size (length dataset))
	 (sum-vec (make-dvec datum-dim 0d0)))
    (doseq (datum dataset)
      (v+ (cdr datum) sum-vec sum-vec))
    (v*n sum-vec (/ 1d0 data-size) sum-vec)
    sum-vec))

(defun standard-deviation-vector (dataset)
  (let* ((datum-dim (length (cdr (elt dataset 0))))
	 (data-size (length dataset))
	 (sum-vec (make-dvec datum-dim 0d0))
	 (ave-vec (mean-vector dataset)))
    (doseq (datum dataset)
      (loop for i from 0 to (1- datum-dim) do
        (let ((diff (- (aref (cdr datum) i) (aref ave-vec i))))
          (incf (aref sum-vec i) (* diff diff)))))
    (loop for i from 0 to (1- datum-dim) do
      (setf (aref sum-vec i) (sqrt (/ (aref sum-vec i) data-size))))
    sum-vec))

;; (defun min-max-vector (training-vector)
;;   (let* ((data-dim (1- (array-dimension (aref training-vector 0) 0)))
;; 	 (data-size (length training-vector))
;; 	 (max-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
;; 	 (min-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
;; 	 (cent-v (make-array data-dim :element-type 'double-float :initial-element 0d0))
;; 	 (scale-v (make-array data-dim :element-type 'double-float :initial-element 0d0)))
;;     ;; init
;;     (loop for j from 0 to (1- data-dim) do
;;       (let ((elem-0j (aref (aref training-vector 0) j)))
;; 	(setf (aref max-v j) elem-0j
;; 	      (aref min-v j) elem-0j)))
;;     ;; calc min-v, max-v
;;     (loop for i from 1 to (1- data-size) do
;;       (loop for j from 0 to (1- data-dim) do
;; 	(let ((elem-ij (aref (aref training-vector i) j)))
;; 	  (when (< (aref max-v j) elem-ij) (setf (aref max-v j) elem-ij))
;; 	  (when (> (aref min-v j) elem-ij) (setf (aref min-v j) elem-ij)))))
;;     ;; calc cent-v, scale-v
;;     (loop for j from 0 to (1- data-dim) do
;;       (setf (aref cent-v j) (/ (+ (aref max-v j) (aref min-v j)) 2.0d0)
;; 	    (aref scale-v j) (if (= (aref max-v j) (aref min-v j))
;; 			       1d0
;; 			       (/ (abs (- (aref max-v j) (aref min-v j))) 2d0))))
;;     (values cent-v scale-v)))

;; (defclass scale-parameters ()
;;   ((centre-vector :accessor centre-vector-of :initarg :centre-vector :type dvec)
;;    (scale-vector :accessor scale-vector-of :initarg :scale-vector :type dvec )))

;; (defun make-scale-parameters (training-vector &key scaling-method)
;;   (cond ((eq scaling-method :unit-standard-deviation)
;; 	 (make-instance 'scale-parameters
;; 	    :centre-vector (mean-vector training-vector)
;; 	    :scale-vector (standard-deviation-vector training-vector)))
;; 	(t
;; 	 (multiple-value-bind (cent-v scale-v)
;; 	     (min-max-vector training-vector)
;; 	   (make-instance 'scale-parameters
;; 	      :centre-vector cent-v
;; 	      :scale-vector scale-v)))))

;; (defun autoscale (training-vector &key scale-parameters)
;;   (let* ((scale-parameters (if scale-parameters scale-parameters
;; 			     (make-scale-parameters training-vector)))
;; 	 (data-dim (1- (array-dimension (aref training-vector 0) 0)))
;; 	 (data-size (length training-vector))	  
;; 	 (new-v (make-array data-size))
;; 	 (cent-v (centre-vector-of scale-parameters))
;; 	 (scale-v (scale-vector-of scale-parameters)))
;;     (loop for i from 0 to (1- data-size) do
;;       (let ((new-data-v (make-array (1+ data-dim) :element-type 'double-float :initial-element 0d0))
;; 	    (data-v (aref training-vector i)))
;; 	(loop for j from 0 to (1- data-dim) do	  
;; 	  (setf (aref new-data-v j) (/ (- (aref data-v j) (aref cent-v j)) (aref scale-v j))))
;; 	(setf (aref new-data-v data-dim) (aref data-v data-dim))
;; 	(setf (aref new-v i) new-data-v)))
;;     (values new-v scale-parameters)))

;; ;; dimension of datum contains the target column so that it can be input of discriminate function. 
;; ;; and target column will be ignored in this function.
;; (defun autoscale-datum (datum scale-parameters)
;;   (let* ((datum-dim (length datum))
;; 	 (new-datum (make-array datum-dim :element-type 'double-float))
;; 	 (cent-v (centre-vector-of scale-parameters))
;; 	 (scale-v (scale-vector-of scale-parameters)))
;;     (loop for i from 0 to (- datum-dim 2) do
;;       (setf (aref new-datum i) (/ (- (aref datum i) (aref cent-v i)) (aref scale-v i))))
;;     (setf (aref new-datum (1- datum-dim)) (aref datum (1- datum-dim)))
;;     new-datum))

;; ;;; Cross-Validation (N-fold)

;; (defun split-training-vector-2part (test-start test-end training-vector sub-training-vector sub-test-vector)
;;     (loop for i from 0 to (1- (length training-vector)) do
;;       (cond ((< i test-start)
;; 	     (setf (aref sub-training-vector i) (aref training-vector i)))
;; 	    ((and (>= i test-start) (<= i test-end))
;; 	     (setf (aref sub-test-vector (- i test-start)) (aref training-vector i)))
;; 	    (t
;; 	     (setf (aref sub-training-vector (- i (1+ (- test-end test-start)))) (aref training-vector i))))))

;; (defun average (list)
;;   (/ (loop for i in list sum i) (length list)))

;; (defun cross-validation (n training-vector kernel &key (c 10) (weight 1.0d0))
;;   (let* ((bin-size (truncate (length training-vector) n))
;; 	 (sub-training-vector (make-array (- (length training-vector) bin-size)))
;; 	 (sub-test-vector (make-array bin-size))
;; 	 (accuracy-percentage-list
;; 	  (loop for i from 0 to (1- n) collect
;; 	    (progn
;; 	      (split-training-vector-2part (* i bin-size) (1- (* (1+ i) bin-size))
;; 					   training-vector sub-training-vector sub-test-vector)
;; 	      (let ((trained-svm (make-svm-model sub-training-vector kernel :c c :weight weight)))
;; 		(multiple-value-bind (useless accuracy-percentage)
;; 		    (svm-validation trained-svm sub-test-vector)
;; 		  (declare (ignore useless))
;; 		  accuracy-percentage))))))
;;     (values (average accuracy-percentage-list)
;; 	    accuracy-percentage-list)))
