#!/bin/env julia

using ArgParse

function parse_commandline()
    s = ArgParseSettings(description = "usage: preprocess_fasta.jl fasta_file --output_dir dir")

    @add_arg_table s begin
        "--fasta_path"
          help = "fasta file path"
          required = true
        "--output_dir"
          help = "output directory"
          required = true
    end
    
    return parse_args(s)
end

function chomp_fasta(fa_file)
    d = Dict{String, String}()
    open(fa_file) do file
        i = 0
        contig = ""
        list = String[]
        while !eof(file)
            i += 1
            i % 1000_000 == 0 && info("preprocessed $i lines of $fa_file ")
            line = readline(file) |> chomp
            if startswith(line, ">chr")
                if length(list) > 0
                    d[contig] = join(list, "")
                end
                contig = line[2:end]
                list = String[]
            else
                push!(list, line)
            end
        end
    end
    d
end

function save_txt(data_dir, d)
    for contig in keys(d)
        if length(contig) > 5
            continue
        end
        s = d[contig]
        l = length(s)
        s_tr = s[1:div(l,2)]
        s_te = s[div(l,2)+1:end]
        write(joinpath(data_dir, string(contig, "_train.txt")), s_tr)
        write(joinpath(data_dir, string(contig, "_test.txt")),  s_te)
    end
end

function preprocess_fasta(data_dir, fa_file)
    d = chomp_fasta(fa_file)
    save_txt(data_dir, d)
end

function main()
    parsed_args = parse_commandline()
    for (arg,val) in parsed_args
        println("$arg => $val")
    end
    data_dir = parsed_args["output_dir"]
    fa_path  = parsed_args["fasta_path"]
    preprocess_fasta(data_dir, fa_path)
end


main()
