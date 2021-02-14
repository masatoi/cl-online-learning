;;; -*- coding:utf-8; mode:lisp -*-

(in-package :clol)

(defmacro define-regression-learner (learner-type (learner input target) &body body)
  `(progn
     (defun ,(intern (catstr (symbol-name learner-type) "-UPDATE"))
         (,learner ,input ,target)
       (declare (type ,learner-type ,learner)
                (type ,(if (sparse-symbol? learner-type)
                         'clol.vector::sparse-vector
                         '(simple-array single-float))
                      ,input)
                (type single-float ,target)
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
  (check-type gamma single-float)
  (assert (<= 0.98 gamma 1.0))
  (%make-rls :input-dimension input-dimension
             :weight (make-vec input-dimension 0.0) ; mu
             :bias 0.0                               ; mu0
             :gamma gamma
             :sigma (make-vec input-dimension 1.0)
             :sigma0 1.0
             :tmp-vec1 (make-vec input-dimension 0.0)
             :tmp-vec2 (make-vec input-dimension 0.0)
             :tmp-float (make-vec 1 0.0)))

(define-regression-learner rls (learner input target)
  (let ((tmp-float (rls-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (f! input (rls-weight learner) (rls-bias learner) tmp-float)
    (let ((loss (- target (aref tmp-float 0)))
          (sigma0 (rls-sigma0 learner))
          (gamma (rls-gamma learner))
          (bias (rls-bias learner)))
      (declare (type single-float loss sigma0 gamma bias))
      (dot! (v* (rls-sigma learner) input (rls-tmp-vec1 learner)) ; Sigma_(t-1) x_t
            input tmp-float) ; x_t^T Sigma_(t-1) x_t
      (let ((beta (/ 1.0 (+ sigma0 (aref tmp-float 0) gamma))))
        (declare (type single-float beta))
        (v*n (rls-tmp-vec1 learner) beta (rls-tmp-vec1 learner)) ; g_t
        (let ((g0 (* beta sigma0)))
          (declare (type single-float g0))        
          ;; Update weight
          (v*n (rls-tmp-vec1 learner) loss (rls-tmp-vec2 learner)) ; delta weight
          (v+ (rls-weight learner) (rls-tmp-vec2 learner) (rls-weight learner))
          ;; Update bias
          (setf (rls-bias learner) (+ bias (* g0 loss)))
          ;; Update sigma
          (v* (rls-tmp-vec1 learner) input (rls-tmp-vec1 learner))
          (v* (rls-tmp-vec1 learner) (rls-sigma learner) (rls-tmp-vec1 learner))
          (v- (rls-sigma learner) (rls-tmp-vec1 learner) (rls-sigma learner))
          (v*n (rls-sigma learner) (/ 1.0 gamma) (rls-sigma learner))
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
  (check-type gamma single-float)
  (assert (<= 0.98 gamma 1.0))
  (%make-sparse-rls :input-dimension input-dimension
                     :weight (make-vec input-dimension 0.0) ; mu
                     :bias 0.0                               ; mu0
                     :gamma gamma
                     :sigma (make-vec input-dimension 1.0)
                     :sigma0 1.0
                     :tmp-vec1 (make-vec input-dimension 0.0)
                     :tmp-vec2 (make-vec input-dimension 0.0)
                     :tmp-float (make-vec 1 0.0)))

(define-regression-learner sparse-rls (learner input target)
  (let ((tmp-float (sparse-rls-tmp-float learner)))
    (declare (type (simple-array single-float 1) tmp-float))
    (sf! input (sparse-rls-weight learner) (sparse-rls-bias learner) tmp-float)
    (let ((index-vector (sparse-vector-index-vector input))
          (loss (- target (aref tmp-float 0)))
          (bias (sparse-rls-bias learner))
          (sigma0 (sparse-rls-sigma0 learner))
          (gamma (sparse-rls-gamma learner)))
      (declare (type (simple-array fixnum) index-vector)
               (type single-float loss bias sigma0 gamma))
      (ds-dot! (ds-v* (sparse-rls-sigma learner) input (sparse-rls-tmp-vec1 learner))  ; Sigma_(t-1) x_t
               input tmp-float) ; x_t^T Sigma_(t-1) x_t
      (let ((beta (/ 1.0 (+ sigma0 (aref tmp-float 0) gamma))))
        (declare (type single-float beta))
        (ps-v*n (sparse-rls-tmp-vec1 learner) beta index-vector (sparse-rls-tmp-vec1 learner)) ; g_t
        (let ((g0 (* beta sigma0)))
          (declare (type single-float g0))
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
          (v*n (sparse-rls-sigma learner) (/ 1.0 gamma) (sparse-rls-sigma learner))
          ;; Update sigma0
          (setf (sparse-rls-sigma0 learner)
                (/ (- sigma0 (* g0 sigma0)) gamma)))))))
