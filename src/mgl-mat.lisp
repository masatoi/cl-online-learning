;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning.mgl-mat
  (:use :cl :hjs.util.vector)
  (:nicknames :clol.mgl-mat)
  (;:export
   ))
   
(in-package :cl-online-learning.mgl-mat)

(ql:quickload :mgl-mat)

(setf mgl-mat:*default-mat-ctype* :float)

(defstruct perceptron-mgl input-dimension weight bias)

;;; Signum function
(defun sign-mgl (x)
  (if (> x 0.0) 1.0 -1.0))

;;; Decision boundary
(defun f-mgl (input weight bias)
  (+ (mgl-mat:dot weight input) bias))

(defparameter mat1 (mgl-mat:make-mat 3 :initial-contents '(1.0 2.0 3.0)))
(defparameter mat2 (mgl-mat:make-mat 3 :initial-contents '(10.0 20.0 30.0)))
(defparameter mat3 (mgl-mat:make-mat 3))

(defun perceptron-mgl-make (input-dimension)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (make-perceptron-mgl :input-dimension input-dimension
			  :weight (mgl-mat:make-mat input-dimension :initial-element 0d0)
			  :bias 0d0))

(defun perceptron-mgl-update (learner input training-label)
  (if (<= (* training-label (f-mgl input
                                   (perceptron-mgl-weight learner)
                                   (perceptron-mgl-bias learner))) 0d0)
    (if (> training-label 0d0)
      (progn
	(mgl-mat:axpy! 1.0 input (perceptron-mgl-weight learner))
	(setf (perceptron-mgl-bias learner) (+ (perceptron-mgl-bias learner) 1d0)))
      (progn
        (mgl-mat:axpy! -1.0 input (perceptron-mgl-weight learner))
        (setf (perceptron-mgl-bias learner) (- (perceptron-mgl-bias learner) 1d0)))))
  learner)

(defun perceptron-mgl-train (learner training-data)
  (etypecase training-data
    (list (dolist (datum training-data)
	    (perceptron-mgl-update learner (cdr datum) (car datum))))
    (vector (loop for datum across training-data do
      (perceptron-mgl-update learner (cdr datum) (car datum)))))
  learner)

(defun perceptron-mgl-predict (learner input)
  (sign-mgl (f-mgl input
                   (perceptron-mgl-weight learner)
                   (perceptron-mgl-bias learner))))

(defun perceptron-mgl-test (learner test-data &key (quiet-p nil))
  (let* ((len (length test-data))
	 (n-correct (count-if (lambda (datum)
				(= (perceptron-mgl-predict learner (cdr datum)) (car datum)))
			      test-data))
	 (accuracy (* (/ n-correct len) 100.0)))
    (if (not quiet-p)
      (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" accuracy n-correct len))
    (values accuracy n-correct len)))

(defparameter a1a-dim 123)
(defparameter a1a-train
  (mapcar (lambda (datum) (cons (car datum) (mgl-mat:array-to-mat (cdr datum))))
          (cl-ol.utils:read-libsvm-data "/home/wiz/tmp/a1a" a1a-dim
                                        :element-type 'single-float)))

(defparameter a1a-test
  (mapcar (lambda (datum) (cons (car datum) (mgl-mat:array-to-mat (cdr datum))))
          (cl-ol.utils:read-libsvm-data "/home/wiz/tmp/a1a.t" a1a-dim
                                        :element-type 'single-float)))

(defparameter perceptron-learner (perceptron-mgl-make a1a-dim))
(time (perceptron-mgl-train perceptron-learner a1a-train))
(time (perceptron-mgl-test perceptron-learner a1a-test))

;;; a9a

(defparameter a9a-dim 123)
(defparameter a9a-train
  (mapcar (lambda (datum) (cons (car datum) (mgl-mat:array-to-mat (cdr datum))))
          (cl-ol.utils:read-libsvm-data "/home/wiz/tmp/a9a" a9a-dim
                                        :element-type 'single-float)))

(defparameter a9a-test
  (mapcar (lambda (datum) (cons (car datum) (mgl-mat:array-to-mat (cdr datum))))
          (cl-ol.utils:read-libsvm-data "/home/wiz/tmp/a9a.t" a9a-dim
                                        :element-type 'single-float)))

(defparameter perceptron-learner (perceptron-mgl-make a9a-dim))
(time (perceptron-mgl-train perceptron-learner a9a-train))
(time (perceptron-mgl-test perceptron-learner a9a-test))

;;; gisette

(defparameter gisette-dim 5000)
(defparameter gisette-train
  (mapcar (lambda (datum) (cons (car datum) (mgl-mat:array-to-mat (cdr datum))))
          (cl-ol.utils:read-libsvm-data "/home/wiz/datasets/gisette_scale" gisette-dim
                                        :element-type 'single-float)))

(defparameter gisette-test
  (mapcar (lambda (datum) (cons (car datum) (mgl-mat:array-to-mat (cdr datum))))
          (cl-ol.utils:read-libsvm-data "/home/wiz/datasets/gisette_scale.t" gisette-dim
                                        :element-type 'single-float)))

(defparameter perceptron-learner (perceptron-mgl-make gisette-dim))
(time (loop repeat 100 do (perceptron-mgl-train perceptron-learner gisette-train)))
(time (perceptron-mgl-test perceptron-learner gisette-test))

;; Evaluation took:
;;   2.918 seconds of real time
;;   2.918665 seconds of total run time (2.914829 user, 0.003836 system)
;;   [ Run times consist of 0.012 seconds GC time, and 2.907 seconds non-GC time. ]
;;   100.03% CPU
;;   9,896,945,702 processor cycles
;;   520,574,176 bytes consed

;; Accuracy: 96.6%, Correct: 966, Total: 1000
;; Evaluation took:
;;   0.012 seconds of real time
;;   0.011698 seconds of total run time (0.011698 user, 0.000000 system)
;;   100.00% CPU
;;   39,964,086 processor cycles
;;   851,600 bytes consed

(defparameter perceptron-learner (perceptron-mgl-make gisette-dim))
(mgl-mat:with-cuda* ()
  (time (loop repeat 100 do (perceptron-mgl-train perceptron-learner gisette-train))))

(time (perceptron-mgl-test perceptron-learner gisette-test))

;; nvcc -arch=sm_50 -I /home/wiz/.roswell/local-projects/cl-cuda/include -ptx -o /tmp/cl-cuda.Hs4gaQ.ptx /tmp/cl-cuda.Hs4gaQ.cu
;; Evaluation took:
;;   16.629 seconds of real time
;;   16.619665 seconds of total run time (15.357693 user, 1.261972 system)
;;   [ Run times consist of 0.025 seconds GC time, and 16.595 seconds non-GC time. ]
;;   99.95% CPU
;;   6 forms interpreted
;;   149 lambdas converted
;;   56,407,194,852 processor cycles
;;   1 page fault
;;   365,484,448 bytes consed

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;; Mini batch ;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

