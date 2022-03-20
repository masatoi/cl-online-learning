;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.utils
  (:use :cl :cl-online-learning.vector)
  (:nicknames :clol.utils)
  (:export :shuffle-vector :read-data
           :defmain :main
           :class-min/max :to-int :to-float))

(in-package :cl-online-learning.utils)

;; Fisher-Yates shuffle
(defun shuffle-vector (vec)
  (loop for i from (1- (length vec)) downto 1 do
    (let* ((j (random (1+ i)))
	   (tmp (svref vec i)))
      (setf (svref vec i) (svref vec j))
      (setf (svref vec j) tmp)))
  vec)

;;; Read dataset

;;; using cl-libsvm-format

(defun read-datum (svmformat-datum data-dimension &key multiclass-p)
  (let* ((input-vector (make-array data-dimension :element-type 'single-float :initial-element 0.0))
         (training-label (if multiclass-p
                             (1- (car svmformat-datum))
                             (coerce (car svmformat-datum) 'single-float))))
    (labels ((iter (data)
               (if (null data)
                   input-vector
                   (progn
                     (setf (aref input-vector (1- (car data)))
                           (coerce (cadr data) 'single-float))
                     (iter (cddr data))))))
      (cons training-label (iter (cdr svmformat-datum))))))

(defun read-datum-sparse (svmformat-datum &key multiclass-p)
  (let* ((training-label (if multiclass-p
                             (1- (car svmformat-datum))
                             (coerce (car svmformat-datum) 'single-float)))
         (sparse-dim (/ (length (cdr svmformat-datum)) 2))
         (input-vector (make-empty-sparse-vector sparse-dim)))
    (labels ((iter (i data)
               (if (null data)
                   input-vector
                   (progn
                     (setf (aref (sparse-vector-index-vector input-vector) i)
                           (1- (car data))
                           (aref (sparse-vector-value-vector input-vector) i)
                           (coerce (cadr data) 'single-float))
                     (iter (1+ i) (cddr data))))))
      (cons training-label (iter 0 (cdr svmformat-datum))))))

(defun read-data (data-path data-dimension &key multiclass-p sparse-p)
  (multiple-value-bind (data-list dim)
      (svmformat:parse-file data-path)
    (let* ((dim (if data-dimension data-dimension dim))
           (reader (if sparse-p
                       (lambda (datum)
                         (read-datum-sparse datum :multiclass-p multiclass-p))
                       (lambda (datum)
                         (read-datum datum dim :multiclass-p multiclass-p)))))
      (values (mapcar reader data-list)
              dim))))

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
	 (sum-vec (make-vec datum-dim 0.0)))
    (doseq (datum dataset)
      (v+ (cdr datum) sum-vec sum-vec))
    (v*n sum-vec (/ 1.0 data-size) sum-vec)
    sum-vec))

(defun standard-deviation-vector (dataset)
  (let* ((datum-dim (length (cdr (elt dataset 0))))
	 (data-size (length dataset))
	 (sum-vec (make-vec datum-dim 0.0))
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
;; 	 (max-v (make-array data-dim :element-type 'single-float :initial-element 0.0))
;; 	 (min-v (make-array data-dim :element-type 'single-float :initial-element 0.0))
;; 	 (cent-v (make-array data-dim :element-type 'single-float :initial-element 0.0))
;; 	 (scale-v (make-array data-dim :element-type 'single-float :initial-element 0.0)))
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
;;       (setf (aref cent-v j) (/ (+ (aref max-v j) (aref min-v j)) 2.0)
;; 	    (aref scale-v j) (if (= (aref max-v j) (aref min-v j))
;; 			       1.0
;; 			       (/ (abs (- (aref max-v j) (aref min-v j))) 2.0))))
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
;;       (let ((new-data-v (make-array (1+ data-dim) :element-type 'single-float :initial-element 0.0))
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
;; 	 (new-datum (make-array datum-dim :element-type 'single-float))
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

;; (defun cross-validation (n training-vector kernel &key (c 10) (weight 1.0))
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

;;; Command line utils

(defun group-arg-list (arg-list)
  (labels ((iter (arg-list req-list key-list)
             (if (null arg-list)
                 (list (reverse req-list) (reverse key-list))
                 (if (char= (aref (car arg-list) 0) #\-)
                     (iter (cddr arg-list) req-list (cons (list (car arg-list) (cadr arg-list)) key-list))
                     (iter (cdr arg-list) (cons (car arg-list) req-list) key-list)))))
    (iter arg-list nil nil)))

(defun split-lambda-list (lambda-list)
  (let ((p (position '&key lambda-list)))
    (if p
        (list (subseq lambda-list 0 p)
              (subseq lambda-list (1+ p)))
        (list lambda-list))))

(define-condition argument-error (simple-error)
  ((argument-error-message :initarg :argument-error-message
                           :initform nil
                           :accessor argument-error-message))
  (:report (lambda (c s)
             (format s "argument-error: ~A" (argument-error-message c)))))

(defun sanity-check (lambda-list arg-list)
  (let ((split-lambda-list (split-lambda-list lambda-list))
        (group-arg-list (group-arg-list arg-list)))
    ;; required key check
    (unless (= (length (car split-lambda-list))
               (length (car group-arg-list)))
      (error (make-condition
              'argument-error
              :argument-error-message (format nil "Incorrect number of required arguments. Required: ~A"
                                              (length (car split-lambda-list))))))
    ;; keyword option check
    (let ((keys (mapcar #'car (cadr split-lambda-list)))
          (argkeys (mapcar (lambda (argkey-pair)
                             (intern (string-upcase (subseq (car argkey-pair) 1))))
                           (cadr group-arg-list))))
      (when (set-difference argkeys keys)
        (error (make-condition
                'argument-error
                :argument-error-message (format nil "Missmatch keyword options. Required: ~A" keys)))))
    (dolist (argkey-pair (cadr group-arg-list))
      (unless (cadr argkey-pair)
        (error (make-condition
                'argument-error
                :argument-error-message "Odd number of keyword arguments."))))))

(defun flatten (structure)
  (cond ((null structure) nil)
        ((atom structure) (list structure))
        (t (mapcan #'flatten structure))))

(defun arg-list->lambda-arg (arg-list)
  (mapcar (lambda (str)
            (if (char= (aref str 0) #\-)
                (intern (string-upcase (subseq str 1)) "KEYWORD")
                str))
          (flatten (group-arg-list arg-list))))

(defmacro defmain (lambda-list &body body)
  (let ((argv (gensym)))
    `(defun main (&rest ,argv)
       ,(if (stringp (car body)) (car body))
       (handler-case
           (progn
             (sanity-check ',lambda-list ,argv)
             (apply (lambda ,lambda-list
                      ,@(if (stringp (car body)) (cdr body) body))
                    (arg-list->lambda-arg ,argv)))
         (argument-error (c)
           (format *error-output* "~A~%" (argument-error-message c))
           (format t "Usage: ~A~%" ',lambda-list)
           (let ((docstr (documentation 'main 'function)))
             (if docstr (format t "~%~A~%" docstr))))
         (error (c)
           (format *error-output* "Error: ~A~%" c)
           (format t "Usage: ~A~%" ',lambda-list)
           (let ((docstr (documentation 'main 'function)))
             (if docstr (format t "~%~A~%" docstr))))))))

(defun class-min/max (read-data-result)
  (let ((min-class most-positive-fixnum)
        (max-class most-negative-fixnum))
    (loop for datum in read-data-result do
      (when (< (car datum) min-class)
        (setf min-class (car datum)))
      (when (> (car datum) max-class)
        (setf max-class (car datum))))
    (list min-class max-class)))

(defun to-int (x)
  (etypecase x
    (float (truncate x))
    (integer x)
    (string (truncate (svmformat::parse-float x)))))

(defun to-float (x)
  (coerce
   (etypecase x
     (float x)
     (integer x)
     (string (svmformat::parse-float x)))
   'single-float))
