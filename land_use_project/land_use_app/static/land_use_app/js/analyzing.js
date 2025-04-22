
        document.addEventListener("DOMContentLoaded", function () {
          const form = document.querySelector("form");
          const basicBtn = document.querySelector('button[name="analysis_type"][value="basic"]');
          const advancedBtn = document.querySelector('button[name="analysis_type"][value="advanced"]');
      
          function hasEmptyFields(formElement) {
            const inputs = formElement.querySelectorAll("input, textarea, select");
            for (let input of inputs) {
              if (
                input.hasAttribute("required") && 
                input.value.trim() === ""
              ) {
                return true;
              }
            }
            return false;
          }
      
          function handleClick(event, button) {
            if (hasEmptyFields(form)) {
              // Prevent the form from submitting
              event.preventDefault();
              alert("Please fill in all required fields.");
            } else {
              button.textContent = "Analyzing...";
            }
          }
      
          basicBtn.addEventListener("click", function (event) {
            handleClick(event, basicBtn);
          });
      
          advancedBtn.addEventListener("click", function (event) {
            handleClick(event, advancedBtn);
          });
        });