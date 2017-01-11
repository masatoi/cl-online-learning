;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.utils
  (:use :cl :cl-online-learning.vector)
  (:nicknames :clol.utils)
  (:export :shuffle-vector
           :read-libsvm-data :read-libsvm-data-multiclass
           :read-libsvm-data-sparse :read-libsvm-data-sparse-multiclass))

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

;; Read libsvm dataset (2-class)
(defun read-libsvm-data (data-path data-dimension &key (element-type 'double-float))
  (let ((data-list nil))
    (with-open-file (f data-path :direction :input)
      (labels ((read-loop (data-list)
		 (let ((read-data (read-line f nil nil)))
		   (if (null read-data)
		     (nreverse data-list)
		     (let* ((dv (make-array data-dimension :element-type element-type :initial-element 0d0))
			    (d (ppcre:split "\\s+" read-data))
			    (index-num-alist
			     (mapcar (lambda (index-num-pair-str)
				       (let ((index-num-pair (ppcre:split #\: index-num-pair-str)))
					 (list (parse-integer (car index-num-pair))
					       (coerce (parse-number:parse-number (cadr index-num-pair)) element-type))))
				     (cdr d)))
			    (training-label (coerce (parse-number:parse-number (car d)) element-type)))
		       (dolist (index-num-pair index-num-alist)
			 (setf (aref dv (1- (car index-num-pair))) (cadr index-num-pair)))
		       (read-loop (cons (cons training-label dv) data-list)))))))
	(read-loop data-list)))))

;; Read libsvm dataset (Multiclass)
(defun read-libsvm-data-multiclass (data-path data-dimension &key (element-type 'double-float))
  (let ((data-list nil))
    (with-open-file (f data-path :direction :input)
      (labels ((read-loop (data-list)
		 (let ((read-data (read-line f nil nil)))
		   (if (null read-data)
		     (nreverse data-list)
		     (let* ((dv (make-array data-dimension :element-type element-type :initial-element 0d0))
			    (d (ppcre:split "\\s+" read-data))
			    (index-num-alist
			     (mapcar (lambda (index-num-pair-str)
				       (let ((index-num-pair (ppcre:split #\: index-num-pair-str)))
					 (list (parse-integer (car index-num-pair))
					       (coerce (parse-number:parse-number (cadr index-num-pair))
                                                       element-type))))
				     (cdr d)))
			    (training-label (1- (parse-integer (car d)))))
		       (dolist (index-num-pair index-num-alist)
			 (setf (aref dv (1- (car index-num-pair))) (cadr index-num-pair)))
		       (read-loop (cons (cons training-label dv) data-list)))))))
	(read-loop data-list)))))

(defun read-libsvm-data-sparse (data-path)
  (let ((data-list nil))
    (with-open-file (f data-path :direction :input)
      (labels ((read-loop (data-list)
		 (let ((read-data (read-line f nil nil)))
		   (if (null read-data)
		     (nreverse data-list)
		     (let* ((d (ppcre:split "\\s+" read-data))
			    (index-num-alist
			     (mapcar (lambda (index-num-pair-str)
				       (let ((index-num-pair (ppcre:split #\: index-num-pair-str)))
					 (list (parse-integer (car index-num-pair))
					       (coerce (parse-number:parse-number (cadr index-num-pair)) 'double-float))))
				     (cdr d)))
			    (training-label (coerce (parse-number:parse-number (car d)) 'double-float))
                            (sparse-dim (length index-num-alist))
                            (sv (make-empty-sparse-vector sparse-dim)))
		       (loop for i from 0 to sparse-dim
                             for index-num-pair in index-num-alist do
                               (setf (aref (sparse-vector-index-vector sv) i) (1- (car index-num-pair))
                                     (aref (sparse-vector-value-vector sv) i) (cadr index-num-pair)))
		       (read-loop (cons (cons training-label sv) data-list)))))))
	(read-loop data-list)))))

(defun read-libsvm-data-sparse-multiclass (data-path)
  (let ((data-list nil))
    (with-open-file (f data-path :direction :input)
      (labels ((read-loop (data-list)
		 (let ((read-data (read-line f nil nil)))
		   (if (null read-data)
		     (nreverse data-list)
		     (let* ((d (ppcre:split "\\s+" read-data))
			    (index-num-alist
			     (mapcar (lambda (index-num-pair-str)
				       (let ((index-num-pair (ppcre:split #\: index-num-pair-str)))
					 (list (parse-integer (car index-num-pair))
					       (coerce (parse-number:parse-number (cadr index-num-pair)) 'double-float))))
				     (cdr d)))
                            (training-label (1- (parse-integer (car d))))
                            (sparse-dim (length index-num-alist))
                            (sv (make-empty-sparse-vector sparse-dim)))
		       (loop for i from 0 to sparse-dim
                             for index-num-pair in index-num-alist do
                               (setf (aref (sparse-vector-index-vector sv) i) (1- (car index-num-pair))
                                     (aref (sparse-vector-value-vector sv) i) (cadr index-num-pair)))
		       (read-loop (cons (cons training-label sv) data-list)))))))
	(read-loop data-list)))))
