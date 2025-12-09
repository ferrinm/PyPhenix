;; Set the Jupyter kernel for all jupyter-python blocks in this project
((jupyter-python-mode . ((org-babel-default-header-args :kernel . "s1_pyphenix"))))

;; tangle source code upon saving
(org-mode . ((eval . (add-hook 'after-save-hook
                               (lambda ()
                                 (when (derived-mode-p 'org-mode)
                                   (org-babel-tangle)))
                               nil t)))))
