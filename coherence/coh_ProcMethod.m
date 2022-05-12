classdef (Abstract) coh_ProcMethod
    % Abstract class for implementing data processing methods.
    %
    % PARAMETERS
    % ----------
    % signal : struct
    % - A struct containing preprocessed data and its associated information,
    %   such as that derived from a python Signal object.
    %
    % verbose : bool; default True
    % - Whether or not to print information about the processing.

    % ===== PROPERTIES =====
    properties(Access=public)
        signal = []
        results = []
        processing_steps = []
        extra_info = []
    end
    properties(Access=protected)
        results_dims_ = []
        verbose = false
        processed = false
        windows_averaged = false
    end
    properties(Dependent)
        results_dims
    end
    
    % ===== METHODS =====
    % --- PUBLIC METHODS ---
    methods
        function obj = coh_ProcMethod(signal, verbose)
            % Initialises the object.
            arguments
                signal (1,:) struct
                verbose (1,1) bool = true
            end
            obj.signal = signal;
            obj.verbose = verbose;
        end

        function dims = get.results_dims(obj)
            % Returns the dimensions of the results, corresponding to the
            % results that will be returned with the 'get_results' method in
            % some subclasses.
            %
            % RETURNS
            % -------
            % dims : array[str]
            % - Dimensions of the results.

            if obj.windows_averaged
                dims = obj.results_dims_(1:end);
            else
                dims = obj.results;
            end
        end
    end

    % --- PRIVATE METHODS ---
    methods (Abstract, Access=protected)
        function obj = sort_inputs(obj)
            % Sorts inputs to the objects.
            obj.processing_steps = obj.signal.processing_steps;
            obj.extra_info = obj.signal.extra_info;
        end

        function optimal_dims = get_optimal_dims(obj)
            % Finds the optimal order of dimensions for the results, following
            % the order ["windows", "channels", "epochs", "frequencies",
            % "timepoints"] based on which dimensions are present in the
            % results.
            %
            % RETURNS
            % -------
            % optimal_dims : array[str]
            % - Optimal dimensions of the results.
            
            possible_order = [
                "windows", "channels", "epochs", "frequencies", "timepoints"
            ];

            optimal_dims = [];
            for dim_i = 1:length(possible_order)
                if ismember(possible_order(dim_i), obj.results_dims)
                    optimal_dims.append(possible_order(dim_i))
                end
            end
        end
    end
end