# Contributing to MindStudio ModelSlim

Thank you for considering contributing to MindStudio ModelSlim. We welcome contributions in any form, whether you are fixing bugs, adding features, improving documentation, or sharing feedback and suggestions. Whether you are an experienced developer or participating in an open-source project for the first time, your help is invaluable.

There are many ways to support the project:

- Report issues or unexpected behavior.
- Suggest or implement new features.
- Improve or expand documentation.
- Review pull requests (PRs) and help other contributors.
- Share the project by introducing MindStudio ModelSlim in blog posts or on social media, or by starring the repository.

We look forward to your participation!

# Finding Issues to Contribute To

If you are looking for a starting point for a new issue, check the following items:

- [good-first-issue](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22797,%22name%22:%22good-first-issue%22%257D%255D,%22label%22:%22good-first-issue%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1)
- [help-wanted](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22796,%22name%22:%22help-wanted%22%257D%255D,%22label%22:%22help-wanted%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1)
- In addition to the two beginner-friendly issue types listed earlier, we also provide other [issue templates](https://gitcode.com/Ascend/msmodelslim/tree/master/.gitcode/ISSUE_TEMPLATE) for reference.
- You can also use [RFC](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%9C%80%E8%BF%91%E5%88%9B%E5%BB%BA%22%257D,%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:25328,%22name%22:%22rfc%22%257D%255D,%22label%22:%22rfc%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1) and [Roadmap](https://gitcode.com/Ascend/msmodelslim/issues?categorysearch=%255B%257B%22field%22:%22labels%22,%22value%22:%255B%257B%22id%22:22807,%22name%22:%22roadmap%22%257D%255D,%22label%22:%22roadmap%22%257D,%257B%22field%22:%22order_by_sort%22,%22value%22:%22created_at_desc%22,%22label%22:%22%E6%8E%92%E5%BA%8F%22%257D%255D&state=all&order_by=created_at&sort=desc&scope=all&page=1) to learn about development plans and roadmaps.

# PRs and Code Reviews

Thank you for submitting a PR. To streamline the review process, please follow these guidelines:

Follow our PR [template and guidelines](https://gitcode.com/Ascend/msmodelslim/blob/master/.gitcode/PULL_REQUEST_TEMPLATE.md).

Refer to the developer documentation [Model Integration Guide](https://gitcode.com/Ascend/msmodelslim/blob/26.0.0/docs/en/developer_guide/integrating_models.md).

For changes that affect user-facing features, update the corresponding user and developer documentation at the same time.

Add or update a test in the continuous integration (CI) workflow. If no test is required, provide the reason.
  
After you finish the preceding preparations, submit the code and enter the `compile` command to trigger the automated build pipeline.

After the pipeline builds successfully, contact the [repository management and maintenance members](https://gitcode.com/Ascend/msmodelslim/member) to request review and merge.

A PR can be merged only after it has all four of the following labels:

   1. `ascend-cla/yes`: CLA check. First-time contributors must sign the CLA. After that, the label is applied automatically to each submission.
   2. `ci-pipeline-passed`: CI pipeline. In the PR process, comment `compile` to trigger it. If the CI checks fail, make the required changes and resubmit.
   3. `lgtm`: Reviewers add this label after approving the PR by commenting `/lgtm`.
   4. `approved`: Committers add this label after approving the PR by commenting `/approved`.

   When your PR has all four labels, it will be merged into the main branch.

## PR Best Practices

- Keep PRs appropriately sized so they are easy to review.
- Each PR should address only one issue or implement one feature.
- Respond to review comments promptly.
- Stay in sync with the main branch and resolve conflicts quickly.

# Building and Testing

After you submit the PR, comment `compile` to trigger the PR pipeline. The platform will automatically run compilation, builds, code checks, and developer tests. If errors occur, fix them according to the messages. If you have questions, contact the [repository management and maintenance members](https://gitcode.com/Ascend/msmodelslim/member).

## PR Title and Category

Only PRs of specific types will be reviewed. Add a proper prefix before your PR title to specify the PR type. The types include:

- `[Feature]`: code related to new features
- `[Bugfix]`: code related to bug fixes
- `[Doc]`: code related to documentation
- `[Test]`: code related to developer tests

## Commit Requirement

To keep the commit history clean, make sure each PR contains only one commit.
If your PR currently contains multiple commits, merge them into a single commit before submitting by using one of the following methods. Although GitCode provides the Squash merge option for PR merging, it is still considered a best practice to sort PRs into a single concise commit in advance.

### Method 1: (Recommended) Interactive Rebase

- View the latest several commits to be merged, for example, the latest three commits.

``` bash
git log --oneline -n 3
```

- Start an interactive rebase, replacing `N` with the number of commits to merge:

``` bash
git rebase -i HEAD~N
```

- In the opened editor:
    - Keep `pick` for the first commit.
    - Change `pick` before the remaining commits to `squash` or `s`.
- Save and close the editor. A new window will open for you to write a concise and meaningful combined commit message.
- Force-push the updated branch, but only to your own feature branch:

``` bash
git push --force-with-lease origin your-branch-name
```

### Method 2: Reset + New Commit

```bash
# Obtain the latest target branch (for example, master) to be merged.
git fetch origin master

# Soft-reset to the main branch. This operation preserves all modifications and moves them back to the staging area.
git reset --soft origin/master

# Commit all changes as a new commit.
git commit -m "feat: concise description of your change"

# Force-push to update the PR branch.
git push --force-with-lease origin your-branch-name

```

> Tip: If you are unsure which target branch to use, check the repository default branch or ask a maintainer.
<br/>
> Warning: Never force-push to a shared or protected branch.

# Acknowledgment

We appreciate your contributions to MindStudio ModelSlim. Every bit of effort makes this project stronger and easier to use. Enjoy creating and coding.
