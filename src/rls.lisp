;;; -*- coding:utf-8; mode:lisp -*-

(in-package :clol)

(defmacro define-regression-learner (learner-type (learner input target) &body body)
  `(progn
     (defun ,(intern (catstr (symbol-name learner-type) "-UPDATE"))
         (,learner ,input ,target)
       (declare (type ,learner-type ,learner)
                (type ,(if (sparse-symbol? learner-type)
                         'clol.vector::sparse-vector
                         '(simple-array double-float))
                      ,input)
                (type double-float ,target)
                (optimize (speed 3) (safety 0)))
       ,@body
       ,learner)
     (defun ,(intern (catstr (symbol-name learner-type) "-TRAIN"))
         (learner training-data)
       (etypecase training-data
         (list (dolist (datum training-data)
                 (,(intern (catstr (symbol-name learner-type) "-UPDATE"))
                   learner (cdr datum) (car datum))))
         (vector (loop for datum across training-data do
           (,(intern (catstr (symbol-name learner-type) "-UPDATE"))
                     learner (cdr datum) (car datum)))))
       learner)
     (defun ,(intern (catstr (symbol-name learner-type) "-PREDICT"))
         (learner input)
       (,(if (sparse-symbol? learner-type) 'sf 'f)
        input
        (,(intern (catstr (symbol-name learner-type) "-WEIGHT")) learner)
        (,(intern (catstr (symbol-name learner-type) "-BIAS")) learner)))
     (defun ,(intern (catstr (symbol-name learner-type) "-TEST"))
         (learner test-data &key (quiet-p nil))
       (flet ((square (x) (* x x)))
         (let* ((len (length test-data))
                (sum-square-error
                  (reduce #'+
                          (mapcar (lambda (datum)
                                    (square (- (,(intern (catstr (symbol-name learner-type) "-PREDICT"))
                                                learner (cdr datum))
                                               (car datum))))
                                  test-data)))
                (rmse (sqrt (/ sum-square-error len))))
           (if (not quiet-p)
               (format t "RMSE: ~A~%" rmse))
           rmse)))))

(defstruct (rls (:constructor %make-rls)
                (:print-object %print-rls))
  input-dimension weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-float)

(defun %print-rls (obj stream)
  (format stream "#S(RLS~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:GAMMA ~A~%~T:SIGMA ~A ...~%~T:SIGMA0 ~A)"
          (rls-input-dimension obj)
          (let ((w (rls-weight obj)))
            (if (> (length w) 10)
                (subseq w 0 10)
                w))
          (rls-bias obj)
          (rls-gamma obj)
          (let ((s (rls-sigma obj)))
            (if (> (length s) 10)
                (subseq s 0 10)
                s))
          (rls-sigma0 obj)))

(defun make-rls (input-dimension gamma)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type gamma double-float)
  (assert (<= 0.98d0 gamma 1d0))
  (%make-rls :input-dimension input-dimension
             :weight (make-dvec input-dimension 0d0) ; mu
             :bias 0d0                               ; mu0
             :gamma gamma
             :sigma (make-dvec input-dimension 1d0)
             :sigma0 1d0
             :tmp-vec1 (make-dvec input-dimension 0d0)
             :tmp-vec2 (make-dvec input-dimension 0d0)
             :tmp-float (make-dvec 1 0d0)))

(define-regression-learner rls (learner input target)
  (let ((tmp-float (rls-tmp-float learner)))
    (declare (type (simple-array double-float 1) tmp-float))
    (f! input (rls-weight learner) (rls-bias learner) tmp-float)
    (let ((loss (- target (aref tmp-float 0)))
          (sigma0 (rls-sigma0 learner))
          (gamma (rls-gamma learner))
          (bias (rls-bias learner)))
      (declare (type double-float loss sigma0 gamma bias))
      (dot! (v* (rls-sigma learner) input (rls-tmp-vec1 learner)) ; Sigma_(t-1) x_t
            input tmp-float) ; x_t^T Sigma_(t-1) x_t
      (let ((beta (/ 1d0 (+ sigma0 (aref tmp-float 0) gamma))))
        (declare (type double-float beta))
        (v*n (rls-tmp-vec1 learner) beta (rls-tmp-vec1 learner)) ; g_t
        (let ((g0 (* beta sigma0)))
          (declare (type double-float g0))        
          ;; Update weight
          (v*n (rls-tmp-vec1 learner) loss (rls-tmp-vec2 learner)) ; delta weight
          (v+ (rls-weight learner) (rls-tmp-vec2 learner) (rls-weight learner))
          ;; Update bias
          (setf (rls-bias learner) (+ bias (* g0 loss)))
          ;; Update sigma
          (v* (rls-tmp-vec1 learner) input (rls-tmp-vec1 learner))
          (v* (rls-tmp-vec1 learner) (rls-sigma learner) (rls-tmp-vec1 learner))
          (v- (rls-sigma learner) (rls-tmp-vec1 learner) (rls-sigma learner))
          (v*n (rls-sigma learner) (/ 1d0 gamma) (rls-sigma learner))
          ;; Update sigma0
          (setf (rls-sigma0 learner)
                (/ (- sigma0 (* g0 sigma0)) gamma)))))))

;;; sparse

(defstruct (sparse-rls (:constructor  %make-sparse-rls)
                        (:print-object %print-sparse-rls))
  input-dimension weight bias
  gamma sigma sigma0 tmp-vec1 tmp-vec2 tmp-float)

(defun %print-sparse-rls (obj stream)
  (format stream "#S(SPARSE-RLS~%~T:INPUT-DIMENSION ~A~%~T:WEIGHT ~A ...~%~T:BIAS ~A~%~T:GAMMA ~A~%~T:SIGMA ~A ...~%~T:SIGMA0 ~A)"
          (sparse-rls-input-dimension obj)
          (let ((w (sparse-rls-weight obj)))
            (if (> (length w) 10)
              (subseq w 0 10)
              w))
          (sparse-rls-bias obj)
          (sparse-rls-gamma obj)
          (let ((s (sparse-rls-sigma obj)))
            (if (> (length s) 10)
              (subseq s 0 10)
              s))
          (sparse-rls-sigma0 obj)))

(defun make-sparse-rls (input-dimension gamma)
  (check-type input-dimension integer)
  (assert (> input-dimension 0))
  (check-type gamma double-float)
  (assert (<= 0.98d0 gamma 1d0))
  (%make-sparse-rls :input-dimension input-dimension
                     :weight (make-dvec input-dimension 0d0) ; mu
                     :bias 0d0                               ; mu0
                     :gamma gamma
                     :sigma (make-dvec input-dimension 1d0)
                     :sigma0 1d0
                     :tmp-vec1 (make-dvec input-dimension 0d0)
                     :tmp-vec2 (make-dvec input-dimension 0d0)
                     :tmp-float (make-dvec 1 0d0)))

(define-regression-learner sparse-rls (learner input target)
  (let ((tmp-float (sparse-rls-tmp-float learner)))
    (declare (type (simple-array double-float 1) tmp-float))
    (sf! input (sparse-rls-weight learner) (sparse-rls-bias learner) tmp-float)
    (let ((index-vector (sparse-vector-index-vector input))
          (loss (- target (aref tmp-float 0)))
          (bias (sparse-rls-bias learner))
          (sigma0 (sparse-rls-sigma0 learner))
          (gamma (sparse-rls-gamma learner)))
      (declare (type (simple-array fixnum) index-vector)
               (type double-float loss bias sigma0 gamma))
      (ds-dot! (ds-v* (sparse-rls-sigma learner) input (sparse-rls-tmp-vec1 learner))  ; Sigma_(t-1) x_t
               input tmp-float) ; x_t^T Sigma_(t-1) x_t
      (let ((beta (/ 1d0 (+ sigma0 (aref tmp-float 0) gamma))))
        (declare (type double-float beta))
        (ps-v*n (sparse-rls-tmp-vec1 learner) beta index-vector (sparse-rls-tmp-vec1 learner)) ; g_t
        (let ((g0 (* beta sigma0)))
          (declare (type double-float g0))
          ;; Update weight
          (ps-v*n (sparse-rls-tmp-vec1 learner) loss index-vector (sparse-rls-tmp-vec2 learner)) ; delta weight
          (dps-v+ (sparse-rls-weight learner) (sparse-rls-tmp-vec2 learner)
                  index-vector (sparse-rls-weight learner))
          ;; Update bias
          (setf (sparse-rls-bias learner) (+ bias (* g0 loss)))
          ;; Update sigma
          (ds-v* (sparse-rls-tmp-vec1 learner) input (sparse-rls-tmp-vec1 learner))
          (dps-v* (sparse-rls-sigma learner) (sparse-rls-tmp-vec1 learner)
                  index-vector (sparse-rls-tmp-vec1 learner))
          (dps-v- (sparse-rls-sigma learner) (sparse-rls-tmp-vec1 learner)
                  index-vector (sparse-rls-sigma learner))
          (v*n (sparse-rls-sigma learner) (/ 1d0 gamma) (sparse-rls-sigma learner))
          ;; Update sigma0
          (setf (sparse-rls-sigma0 learner)
                (/ (- sigma0 (* g0 sigma0)) gamma)))))))

;; ;;; example

;; (defparameter rls1 (make-rls 1 0.9d0))

;; (ql:quickload :wiz-util)
;; (ql:quickload :clgplot)

;; (defparameter x-lst
;;   (wiz:n-times-collect 100
;;     (make-array 1 :element-type 'double-float :initial-element (wiz:random-uniform (- pi) pi))))

;; (defparameter y-lst
;;   (mapcar (lambda (x)
;;             (+ (sin (aref x 0)) (wiz:random-normal :sd 0.1d0) 1d0))
;;           x-lst))

;; (defparameter train-dataset
;;   (mapcar #'cons y-lst x-lst))

;; (train rls1 train-dataset)

;; (defparameter x-lst-test
;;   (loop for x from (- pi) to pi by 0.1d0 collect
;;         (make-array 1 :element-type 'double-float :initial-element x)))

;; (defparameter y-lst-test
;;   (mapcar (lambda (x)
;;             (+ (sin (aref x 0)) (wiz:random-normal :sd 0.1d0)))
;;           x-lst-test))

;; (rls-test rls1 (mapcar #'cons y-lst-test x-lst-test))

;; (clgp:plots
;;  (list y-lst
;;        (loop for x in x-lst-test
;;              collect (rls-predict rls1 x)))
;;  :x-seqs (list (mapcar (lambda (x) (aref x 0)) x-lst)
;;                (wiz:seq (- pi) pi :by 0.1d0))
;;  :style '(points lines)
;;  :output "/home/wiz/tmp/rls.png")

;; (defparameter x-lst-sparse
;;   (mapcar (lambda (x)
;;             (make-sparse-vector (make-array 1 :element-type 'fixnum :initial-element 0) x))
;;           x-lst))

;; (defparameter train-dataset-sparse
;;   (mapcar #'cons y-lst x-lst-sparse))

;; (defparameter sparse-rls1 (make-sparse-rls 1 0.9d0))
;; (train sparse-rls1 train-dataset-sparse)

;; (defparameter x-lst-test-sparse
;;   (loop for x from (- pi) to pi by 0.1d0
;;         collect (make-sparse-vector (make-array 1 :element-type 'fixnum :initial-element 0)
;;                                     (make-array 1 :element-type 'double-float :initial-element x))))

;; (clgp:plots
;;  (list y-lst
;;        (loop for x in x-lst-test
;;              collect (rls-predict rls1 x))
;;        (loop for x in x-lst-test-sparse
;;              collect (sparse-rls-predict sparse-rls1 x)))
;;  :x-seqs (list (mapcar (lambda (x) (aref x 0)) x-lst)
;;                (wiz:seq (- pi) pi :by 0.1d0)
;;                (wiz:seq (- pi) pi :by 0.1d0))
;;  :style '(points lines lines)
;; ; :output "/home/wiz/tmp/rls.png"
;;  :title-list '("training-data" "RLS(dense)" "RLS(sparse)")
;;  )
