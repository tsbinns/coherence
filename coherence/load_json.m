function contents = load_json(fpath, verbose)
    % Loads the content of a json file into a struct.
    %
    % PARAMETERS
    % ----------
    % fpath : string
    % - Filepath to the location of the json file to load, optionally including
    %   the ".json" filetype.
    %
    % verbose : logical
    % - Whether or not to print information about the filepath of the file being
    %   loaded.
    %
    % RETURNS
    % -------
    % contents : struct
    % - Contents of the json file loaded into a struct

    arguments
        fpath {mustBeA(fpath, "char")}
        verbose {mustBeA(verbose, "logical")} = true
    end

    if fpath(length(fpath)-4:end) ~= ".json"
        fpath = fpath + ".json";
    end

    if verbose
        fprintf("Loading contents from the json file:\n%s\n", fpath)
    end

    file = fopen(fpath);
    file_data = fread(file, inf);
    data_as_str = char(file_data');
    fclose(file);
    contents = jsondecode(data_as_str);

end