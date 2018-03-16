# How to contribute

Contributions and bug fixes are welcomed. Here's how to get involved.

## General guidelines

* The code must obey a given format (specified by the `.clang-format` file in
  the top-level directory).  To impose this format, you can run `clang-format
  -i *.h *.cc` on any files you touch, or set up almost any text editor to do
  this automatically.  See the
  [documentation](https://clang.llvm.org/docs/ClangFormat.html) for more
  information.

* Direct pushes to the main repository are not permitted.  This allows sanity
  checks to be performed before merging any changes.  This is effected by
  mandating changes be provided in the form of pull requests, for which the
  procedure is outlined below.

* In the spirit of clean and readable code, please also keep commit messages
  informative and consistent.  A very short and excellent set of guidelines,
  with examples, can be found [here](https://chris.beams.io/posts/git-commit/);
  please read before getting stuck into writing code for this project.

* Don't be afraid of breaking things!  Because you'll be pushing requests, any
  changes will be subjected to testing and peer review before merging.  If we
  don't like what you've done, we'll either guide you through any changes we'd
  like, or we'll simply reject the changes if they're not deemed appropriate.
  Either way, you can't break anything in the code without us having a chance
  to intervene.

## Process for contributing

* Add some really awesome code to your local fork [on a branch other than
  master](http://blog.jasonmeridth.com/posts/do-not-issue-pull-requests-from-your-master-branch/).
  Try to give the [new
  branch](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
  a name related to the change you're going to make.
* When you are ready for others to examine and comment on your new feature,
  navigate to your fork of Psi4 on GitHub and open a [pull
  request](https://help.github.com/articles/using-pull-requests/) (PR). Note
  that after you launch a PR from one of your fork's branches, all subsequent
  commits to that branch will be added to the open pull request automatically.
  Each commit added to the PR will be validated for mergability, compilation
  and test suite compliance; the results of these tests will be visible on the
  PR page.
* If you're providing a new feature, you must add test cases and documentation.
* When the code is ready to go, make sure you run the full or relevant portion
  of the test suite on your local machine to check that nothing is broken.
* When you're ready to be considered for merging, check the "Ready to go"
  box on the PR page to let the team know that the changes are complete.  The
  code will not be merged until this box is checked, the continuous integration
  tests have been completed, and core developers have reviewed the changes.

# Additional Resources

* [General GitHub documentation](https://help.github.com/)
* [PR best practices](http://codeinthehole.com/writing/pull-requests-and-other-good-practices-for-teams-using-github/)
* [A guide to contributing to software packages](http://www.contribution-guide.org)
* [Thinkful PR example](http://www.thinkful.com/learn/github-pull-request-tutorial/#Time-to-Submit-Your-First-PR)

