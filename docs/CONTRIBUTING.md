#### General
- Before fill an issue browse trough [Documentation](https://kismuz.github.io/btgym/), 
project [Wiki](https://github.com/Kismuz/btgym/wiki) 
and [existing issues](https://github.com/Kismuz/btgym/issues?utf8=%E2%9C%93&q=) 
to make sure  your question, proposal or error to be reported has neither has been answered before
 nor is already under development / in process of beenig fixed.
 
 
- Make sure you have latest version of package installed. 
- If you find a Closed issue that seems like it is the same or similar topic, 
  open a new issue and include a link to the original issue in the body of your new one.
- Use a clear and descriptive title for the issue to identify the problem or suggestion.
- Use tags to categorize your issue.
- Be sure to use @mention if you expect response from specific user.


#### Reporting bugs and errors
Include as many details as possible:

- your running environmnet (OS name and version, python and related packages versions);
- files or package parts has been run;
- any modifications made to code provided with package;
- expected behaviour;
- actiual behaviour;
- any specific steps to reproduce behaviour;
- relevant log output;
- error traceback;
- relevant screenshots;

*Note*:
- BTgym algorithms framework uses distributed Tensorflow training setup. Make sure to set suited 
  level of verbosity to correctly identify primiry exception:
 
   ```
        verbose kwarg(int):
        0 - Notices and Errors (suitable for normal execution),
        1 - Info level (checking inner dynamics), 
        2 - Debugging level (excessive logging output). 
   ```

#### Proposing framework features:
- Provide a step-by-step description and motivation of the suggested feature or enhancement 
  in as many details as possible;
- Provide specific examples to demonstrate the steps; Include copy/pasteable 
  snippets which you use in those examples as Markdown code blocks.
- Describe the current behavior and explain which behavior you expected to see 
  instead and why;
- Explain why this enhancement would be useful;


#### Asking general questions
- BTGym is small research project and it's absolutely ok to open an issue to ask a question, 
propose a feature or open a research-related discussion.

#### Contributing to project
- Follow general GitHub guidelines when preparing pull requests:

https://help.github.com/categories/collaborating-with-issues-and-pull-requests/

https://gist.github.com/Chaser324/ce0505fbed06b947d962





