#|
  This file is a part of cl-online-learning project.
|#

(in-package :cl-user)
(defpackage cl-online-learning-test-asd
  (:use :cl :asdf))
(in-package :cl-online-learning-test-asd)

(defsystem cl-online-learning-test
  :author ""
  :license ""
  :depends-on (:cl-online-learning
               :rove)
  :components ((:module "t"
                :components
                ((:file "cl-online-learning"))))
  ;; Two ways a rove run can be green without being good, so check for both.
  ;; ROVE:RUN reports failure by its return value rather than by signalling, so
  ;; without the UNLESS a red suite would leave ASDF:TEST-SYSTEM exiting 0.  And
  ;; it reports success when it finds no suites at all, so a run that executed
  ;; nothing is indistinguishable from a passing one -- observed on ccl-bin in
  ;; CI, where `rove <asd>' printed not one assertion and still exited 0.
  :perform (test-op (o c)
             (let ((tests (loop for suite in (uiop:symbol-call :rove/core/suite/package
                                                               :system-suites c)
                                when suite
                                  append (uiop:symbol-call :rove/core/suite/package
                                                           :suite-tests suite))))
               (unless tests
                 (error "No rove tests are registered for ~A -- the suite did not load, ~
                         so nothing ran."
                        (asdf:component-name c)))
               (unless (uiop:symbol-call :rove :run c)
                 (error "Tests failed.")))))
