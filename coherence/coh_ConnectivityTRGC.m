classdef coh_ConnectivityTRGC < coh_ProcMethod
    % Calculates the time-reversed Granger causality (TRGC) between signals.
    %
    % PARAMETERS
    % ----------
    % signal : struct
    % - The preprocessed data to analyse.
    %
    % verbose : bool; default false
    % - Whether or not to print information about the processing.
    %
    % METHODS
    % -------
    % process
    % - Performs TRGC analysis.
    %
    % save_results
    % - Converts the results and additional information to a struct, and saves
    %   this as a file.
    %
    % results_as_struct
    % - Converts the results and additional information to a struct, and returns
    %   this.

    properties
        Property1
    end

    methods
        function obj = untitled3(inputArg1,inputArg2)
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = inputArg1 + inputArg2;
        end

        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end