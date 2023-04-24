$(document).ready( function() {
    var genRandomDigits = function(N){
      var container = Array();
      for (let i = 0; i < 2; i++) {
        var foo = Array(N).fill(0).map((e,i)=>Math.floor(Math.random() * 10))
        var fooMod = foo.toString().replace(/,/g, '');
        container.push(fooMod)
      }
      return container;
    };

    var genRandomID = function(N) {
        let result = '';
        const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        const charactersLength = characters.length;
        let counter = 0;
        while (counter < N) {
          result += characters.charAt(Math.floor(Math.random() * charactersLength));
          counter += 1;
        }
        return result;
    }

    // Display service task when clicking on Run
    $('.service_menu').on('click', 'button[class="run"]', function (e) {
      // Define variables
      var serviceType = $(this).closest('.service_menu').attr('class').split(' ')[1];
      var thisServiceMenu = '.service_menu.' + serviceType
      var thisServiceResult = '.service_results.' + serviceType
      var currentdate = new Date()
      var datetime = String(currentdate.getDate()).padStart(2, '0') + "."
                      + String(currentdate.getMonth()+1).padStart(2, '0')  + "."
                      + currentdate.getFullYear() + " "
                      + String(currentdate.getHours()).padStart(2, '0') + ":"
                      + String(currentdate.getMinutes()).padStart(2, '0') + ":"
                      + String(currentdate.getSeconds()).padStart(2, '0');
      var div = $(
        '<div class="task active">' +
        '<div class="date"></div>' +
        '<div class="name"></div>' +
        '<div class="delete">' +
        '<button type="button" class="delete_button">x</button></div>' +
        '<div class="results">' +
        '<button type="submit" class="panel_button">Panel</button></div>' +
        '<div class="progress"></div>' +

        '<div class="panel_menu" style="display:none;">' +

        '<div class="panel_choice"><span>Results</span><span class="symbol"> &#9660;</span>' +
        '<div class="panel_submenu">' +
        '<a class="panel_submenu_config" value="Config" href="javascript:void(0)">Config</a>' +
        '<a class="panel_submenu_results" value="Results" href="javascript:void(0)">Results</a>' +
        '</div></div>' +

        '<div class="results_menu_content" style="display:block;">' +
        '<input type="checkbox" name="autoscroll" value="on" checked>' +
        '<label for="checkbox">autoscroll</label>' +
        '<button type="button" class="clear_logs" value="clear">clear</button></div></div>' +

        '<div class="panel_content" style="scroll-behavior:smooth;max-height:430px; max-width: 570px; min-width:570px;overflow: auto;resize:both;display: none;height:120px;overflow:auto;">' +
        '<span class="results_to_display" style="display: block;"></span>' +
        '<span class="config_to_display" style="display: none"></span>' +
        '</div></div>');

      // Generate random ID for results log block
      var taskID = genRandomID(15);

      // Get task name from form
      var randomTaskId = "task-deepmodel-"+serviceType+"-"+taskID;
      var userTaskName = $(thisServiceMenu).find("input[name='task_name']").val();
      var stringLength = 28
      var trimmedString = userTaskName.length < 1 ? randomTaskId : userTaskName;
      var trimmedString = trimmedString.length > stringLength ?
                        trimmedString.substring(0, stringLength - 3) + "..." : trimmedString;

      // Assign attributes to added elements
      $(div[0]).attr('id', randomTaskId);
      $(div[0].childNodes[0]).text(datetime);
      $(div[0].childNodes[1]).text(trimmedString);
      $(div[0]).find('button[class="delete_button"]').attr('id', "task-deepmodel-"+serviceType+"-delete-" + taskID);
      $(div[0]).find('button[class="delete_button"]').attr('disabled', true);
      $(div[0].childNodes[4]).text("ACTIVE");
      $(div[0].childNodes[4]).attr('id', "task-deepmodel-"+serviceType+"-progress-" + taskID);
      $(div[0].childNodes[5]).attr('id', "task-deepmodel-"+serviceType+"-logs_menu-" + taskID);
      $(div[0].childNodes[6]).attr('id', "task-deepmodel-"+serviceType+"-logs-" + taskID);
      $(div[0]).find('button[class="panel_button"]').attr('id', "task-deepmodel-"+serviceType+"-results-" + taskID);
      $(div[0]).find('button[class="clear_logs"]').attr('id', "task-deepmodel-"+serviceType+"-clear_logs-" + taskID);
      $(div[0]).find('input[name="autoscroll"]').attr('id', "task-deepmodel-"+serviceType+"-autoscroll-" + taskID);

      // Limit number of tasks shown
      if ($(thisServiceResult).children().length > 4){
        $(thisServiceResult).children().last().remove();
      }

      // Add html
      $(thisServiceResult).prepend(div).after();

      // Toggle run-cancel buttons
      $(this).attr('disabled', true);

      // Write service parameters to config panel
      var configContent = $(thisServiceResult).children().first().find(".config_to_display");

      configContent.append('<p>USER INPUTS</p>');
      configContent.append('<p>- - - - - - - - - - - - - - - - - - - - - - - - - -</p>');

      // Add service parameters
      $(thisServiceMenu).find('form[method="POST"]').children('input').each(function(i, v) {
        if (v.type != 'checkbox'){
            if (v.value == 0){
                configContent.append('<p>'+v.name+': <span style="color:#59b585"><i>&lt;none&gt;</i></span></p>')
            } else {
                configContent.append('<p>'+v.name+': <span style="color:#4674b0">'+v.value+'</span></p>')
            }
        } else {
            if (v.checked == true){
                configContent.append('<p>'+v.name+': <span style="color:#4674b0">included</span></p>')
            } else {
                configContent.append('<p>'+v.name+': <span style="color:#4674b0">not_included</span></p>')
            }
        }
      });

      // Add model info from select menu
      var selectName = $(thisServiceMenu).find('select[class="models"]').attr('name');
      $(thisServiceMenu).find('select[class="models"]').children('option').each(function(i, v){
          if (v.selected){
              if (v.value == 'None'){
                  configContent.append('<p>'+selectName+': <span style="color:#59b585"><i>&lt;none&gt;</i></span></p>')
              } else {
                  configContent.append('<p>'+selectName+': <span style="color:#4674b0">'+v.value+'</span></p>')
              }
          }
      });

      // Prevent reload page
      e.preventDefault();
    });

    // Cancel actions
    $('.service_menu').delegate('button[class="cancel"]', 'click', function (e) {
      // Define variables
      var serviceType = $(this).closest('.service_menu').attr('class').split(' ')[1];
      var thisServiceMenu = '.service_menu.' + serviceType
      var thisServiceResult = '.service_results.' + serviceType

      // Toggle run-cancel buttons
      $(thisServiceMenu).find('button[class="run"]').removeAttr('disabled');

      // Change task status
      var currentTaskClass = $(thisServiceResult).children().first().attr('class')
      var currentTaskStatus = currentTaskClass.split(' ')[1];
      if(currentTaskStatus == "active"){
        $(thisServiceResult).children().first().removeClass('active').addClass('cancelled');
        $(thisServiceResult).children().first().find('.progress').text('STOPPED');
      }

      // Make delete button available
      var currentTaskID = $(thisServiceResult).children().first().attr('id').split('-')[3]
      $("#task-deepmodel-"+serviceType+"-delete-"+currentTaskID).removeAttr("disabled");

      // Prevent reload page
      e.preventDefault();
    });

    // Results actions
    $('.service_block').delegate('button[class="panel_button"]', 'click', function (e) {
        // Define variables
      var serviceType = $(this).attr('id').split('-')[2];
      var taskID = $(this).attr('id').split('-')[4];

      // Change background color
      var color = $("#"+"task-deepmodel-"+serviceType+"-"+taskID).css('background-color');
      if(color == "rgba(0, 0, 0, 0)"){
        $('#'+"task-deepmodel-"+serviceType+"-"+taskID).css('background-color', 'white');
      } else {
        $('#'+"task-deepmodel-"+serviceType+"-"+taskID).css('background-color', "rgba(0, 0, 0, 0)");
      }

      // Toggle result logs
      $('#'+"task-deepmodel-"+serviceType+"-logs-"+taskID).toggle();
      $('#'+"task-deepmodel-"+serviceType+"-logs_menu-"+taskID).toggle();

      // Prevent reload page
      e.preventDefault();
    });

    // Delete actions
    $('.service_block').delegate('button[class="delete_button"]', 'click', function (e) {
      // Define variables
      var serviceType = $(this).attr('id').split('-')[2];
      var taskID = $(this).attr('id').split('-')[4];

      // Toggle result logs
      $('#'+"task-deepmodel-"+serviceType+"-"+taskID).remove();

      // Prevent reload page
      e.preventDefault();
    });

    // Clear logs actions
    $('.service_results').delegate('button[class="clear_logs"]', 'click', function (e) {
      // Define variables
      var serviceType = $(this).attr('id').split('-')[2];
      var taskID = $(this).attr('id').split('-')[4];

      // Clear result logs
      $('#'+"task-deepmodel-"+serviceType+"-logs-"+taskID).find('.results_to_display').empty();
    });

    // Panel submenu actions
    $('.service_block').delegate('.panel_submenu a', 'click', function (e) {
      // Define variables
      var selectedText = $(this).text();
      var menuText = $(this).closest('.panel_choice').find('span').first().text();
      var serviceType = $(this).closest('.panel_menu').attr('id').split('-')[2];
      var taskID = $(this).closest('.panel_menu').attr('id').split('-')[4];

      // Replace text
      $(this).closest('.panel_choice').find('span').first().text(selectedText);

      if (selectedText == "Config"){
        // Toggle config-results
        $(this).closest('.panel_menu').find('.results_menu_content').css({'display': 'none'});
        $('#task-deepmodel-'+serviceType+'-logs-'+taskID).find('.results_to_display').css({'display': 'none'});
        $('#task-deepmodel-'+serviceType+'-logs-'+taskID).find('.config_to_display').css({'display': 'block'});
      } else {
        // Show results submenu
        $(this).closest('.panel_menu').find('.results_menu_content').css({'display': 'block'});
        $('#task-deepmodel-'+serviceType+'-logs-'+taskID).find('.results_to_display').css({'display': 'block'});
        $('#task-deepmodel-'+serviceType+'-logs-'+taskID).find('.config_to_display').css({'display': 'none'});
      }
    });

    // ---------------------- FORM: PROCESS INPUT VALUES & SUBMIT VIA AJAX
    $('.service_menu').on('click', 'button', function (e) {
        // Specify service block considered
        var serviceType = $(this).closest('.service_menu').attr('class').split(' ')[1];
        var thisServiceMenu = '.service_menu.' + serviceType
        var thisServiceBlock = '.service_block.' + serviceType

        // Get form data (+task id and submit button class)
        const container = {
          submit_button: $(this).attr('class'),
          task_id: $(thisServiceBlock).find('.service_results').children(":first").attr('id'),
        };

        // process input fields
        $(thisServiceMenu).find('form[method="POST"]').children('input').each(function(i, v) {
          if (v.type == 'checkbox'){
            if (v.checked == true){
              container[v.name] = true;
            } else {
              container[v.name] = false;
            }
          } else {
            if (v.type == 'number'){
              container[v.name] = parseInt(v.value, 10);
            } else {
              container[v.name] = v.value;
            }
          }
        });

        // process selection menu items
        var selectName = $(thisServiceMenu).find('select[class="models"]').attr('name');
        $(thisServiceMenu).find('select[class="models"]').children('option').each(function(i, v){
          if (v.selected){
            if (v.value == 'None'){
              container[selectName] = '';
            } else {
              container[selectName] = v.value;
            }
          }
        });

        // Empty form inputs
        var buttonClass = $(this).attr('class')
        if(buttonClass == 'run'){
          // Set inputs to None
          $(thisServiceMenu).find("input").each(function(i, v){
            if (v.type == 'number'){
              // If input type is number, set to placeholder value
              var placeholderValue = $(this).attr('placeholder');
              $(this).val(placeholderValue);
            } else {
              $(this).val('');
            }
          })
          // Set model selection to None
          $(thisServiceMenu).find('select[class="models"]').val("None").change();
        }

        // Make request
        $.ajax({
          url: "http://127.0.0.1:5051/"+serviceType,
          data: JSON.stringify(container),
          type: "POST",
          contentType: false,
          processData: false,
          cache: false
        });

        // Prevent reload page
        e.preventDefault();
    });

    // ---------------------- SELECT MENU: ADD MODEL NAMES
    var getFilesFolders = function(container, address){
        var tmp = null;
        $.ajax({
              url: address,
              data: JSON.stringify(container),
              dataType: "json",
              method: "POST",
              async: false,
              success: function(data){
                tmp = data;
              },
              error: function(request, status, error){
                alert('Error: '+error);
              }
            });
        return tmp;
    };

    // ----- Make request: Models to restore
    response = getFilesFolders(
        {folder_path: "results/weights"},
        "http://127.0.0.1:5051/getfiles"
        );

    // Feed select options
    for (val of response.folders) {
        var item = $('<option value="'+val+'">'+val+'</option>');
        $('select.models').children('option[value="None"]').after(item);
    };
});
